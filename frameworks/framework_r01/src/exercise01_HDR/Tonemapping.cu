#include "Tonemapping.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>

#include "opg/hostdevice/misc.h"

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ

namespace {

    // Implementation of atomicMax function for float
    __device__ static float atomicMax(float* address, float val)
    {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    // Implementation of atomicMin function for float
    __device__ static float atomicMin(float* address, float val)
    {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    // Converts a single HSV color to RGB space
    __device__ glm::vec3 hsvToRgb(glm::vec3 pixel)
    {
        float r, g, b;
        float h, s, v;
        
        h = pixel.x;
        s = pixel.y;
        v = pixel.z;
        
        float f = h/60.0f;
        float hi = glm::floor(f);
        f = f - hi;
        float p = v * (1 - s);
        float q = v * (1 - s * f);
        float t = v * (1 - s * (1 - f));
        
        if (hi == 0.0f || hi == 6.0f)
        {
            r = v;
            g = t;
            b = p;
        }
        else if(hi == 1.0f)
        {
            r = q;
            g = v;
            b = p;
        }
        else if (hi == 2.0f)
        {
            r = p;
            g = v;
            b = t;
        }
        else if (hi == 3.0f)
        {
            r = p;
            g = q;
            b = v;
        }
        else if (hi == 4.0f)
        {
            r = t;
            g = p;
            b = v;
        }
        else
        {
            r = v;
            g = p;
            b = q;
        }
        
        return glm::vec3(r, g, b);
    }
    
    // Converts a single RGB color to HSV space
    __device__ glm::vec3 rgbToHsv(glm::vec3 pixel)
    {
        float r, g, b;
        float h, s, v;

        r = pixel.x;
        g = pixel.y;
        b = pixel.z;
        
        float max = glm::compMax(pixel);
        float min = glm::compMin(pixel);
        float diff = max - min;
        
        v = max;
        
        if(v == 0.0f) // black
        {
            h = s = 0.0f;
        }
        else
        {
            s = diff / v;
            if(diff < 0.001f) // grey
            {
                h = 0.0f;
            }
            else // color
            {
                if(max == r)
                {
                    h = 60.0f * (g - b)/diff;
                    if(h < 0.0f) { h += 360.0f; }
                }
                else if (max == g)
                {
                    h = 60.0f * (2 + (b - r)/diff);
                }
                else
                {
                    h = 60.0f * (4 + (r - g)/diff);
                }
            }		
        }
        
        return glm::vec3(h, s, v);
    }
}

__global__ void extractRgbFromMultiChannelKernel(float* hdr_in, glm::vec3* hdr_rgb_out, uint32_t channels, uint32_t number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    glm::vec3 hdr_rgb = *reinterpret_cast<glm::vec3*>(hdr_in + channels * gid);
    hdr_rgb_out[gid] = hdr_rgb;
}

__global__ void convertFloatToUint8Kernel(float* hdr, uint8_t* ldr, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    float c = hdr[gid];
    c = glm::clamp(c, 0.0f, 1.0f);
    // sRGB gamma correction is already applied in tonemapping exercise
    // c = powf(c, 1.0f / gamma) * 255.0f + 0.5f;
    // c = glm::clamp(c, 0.0f, 255.0f);
    c = c * 255.0f;
    ldr[gid] = static_cast<uint8_t>(c);
}

__global__ void maxKernel(float* values, float* max_value, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    atomicMax(max_value, values[gid]);
}

__global__ void minKernel(float* values, float* min_value, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    atomicMin(min_value, values[gid]);
}

__global__ void brightnessMinMaxKernel(glm::vec3* hsv_values, float* min_value, float* max_value, uint32_t number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    float v = hsv_values[gid].z;
    atomicMin(min_value, v);
    atomicMax(max_value, v);
}

__global__ void boxFilterRgbKernel(glm::vec3 *hdr_rgb_in, glm::vec3 *hdr_rgb_out, uint32_t filter_size, uint32_t width, uint32_t height)
{
    const glm::uvec3 gid = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);
    if (gid.x >= width || gid.y >= height)
    {
        return;
    }

    uint32_t row = gid.y;
    uint32_t col = gid.x;

    glm::vec3 hdr_rgb_acc = glm::vec3(0);
    for (uint32_t i = 0; i < filter_size; i++)
    {
        for (uint32_t j = 0; j < filter_size; j++)
        {
            uint32_t eff_row = glm::clamp(row + i - filter_size / 2, 0u, height - 1);
            uint32_t eff_col = glm::clamp(col + j - filter_size / 2, 0u, width - 1);
            uint32_t idx = eff_row * width + eff_col;
            // uint32_t idx = eff_col * width + eff_row;
            hdr_rgb_acc += hdr_rgb_in[idx];
        }
    }

    float filter_elements = static_cast<float>(filter_size * filter_size);
    uint32_t out_idx = row * width + col;
    hdr_rgb_out[out_idx] = hdr_rgb_acc / filter_elements;
}

__global__ void convertRgbToHsvBrightnessKernel(glm::vec3* rgb_values, glm::vec3* hsv_values, uint32_t number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    glm::vec3 p_rgb = rgb_values[gid];
    glm::vec3 p_hsv = rgbToHsv(p_rgb);
    p_hsv.z = glm::max(p_hsv.z, 1e-4f);
    p_hsv.z = glm::log(p_hsv.z);
    
    hsv_values[gid] = p_hsv;
}

__global__ void convertHsvToRgbKernel(glm::vec3* hsv_values, glm::vec3* rgb_values, uint32_t number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    glm::vec3 p_hsv = hsv_values[gid];
    glm::vec3 p_rgb = hsvToRgb(p_hsv);
    
    rgb_values[gid] = p_rgb;
}

// TODO: put your CUDA kernels and the host functions which launch the kernels here
//

void maxValue(float* values, float* max_value, uint32_t number_values)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_values, block_size ); // Spawn enough blocks
    maxKernel<<<block_count, block_size>>>(values, max_value, number_values);
}

void minValue(float* values, float* min_value, uint32_t number_values)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_values, block_size ); // Spawn enough blocks
    minKernel<<<block_count, block_size>>>(values, min_value, number_values);
}

void brightnessMinMax(glm::vec3* hsv_values, float* min_value, float* max_value, uint32_t number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    brightnessMinMaxKernel<<<block_count, block_size>>>(hsv_values, min_value, max_value, number_pixels);
}

void boxFilterRgb(glm::vec3 *hdr_rgb_in, glm::vec3 *hdr_rgb_out, uint32_t filter_size, uint32_t width, uint32_t height)
{
    // launch kernel
    const glm::uvec3 block_size  = glm::uvec3(8, 8, 1); // 512 is a size that works well with modern GPUs.
    const glm::uvec3 block_count = ceil_div(glm::uvec3(width, height, 1), block_size ); // Spawn enough blocks
    boxFilterRgbKernel<<<glm2cuda(block_count), glm2cuda(block_size)>>>(hdr_rgb_in, hdr_rgb_out, filter_size, width, height);
}

void convertRgbToHsvBrightness(glm::vec3* rgb_values, glm::vec3* hsv_values, uint32_t number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    convertRgbToHsvBrightnessKernel<<<block_count, block_size>>>(rgb_values, hsv_values, number_pixels);
}

void convertHsvToRgb(glm::vec3* hsv_values, glm::vec3* rgb_values, uint32_t number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    convertHsvToRgbKernel<<<block_count, block_size>>>(hsv_values, rgb_values, number_pixels);
}

void extractRgbFromMultiChannel(float* hdr_in, glm::vec3* hdr_rgb_out, uint32_t channels, uint32_t number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    extractRgbFromMultiChannelKernel<<<block_count, block_size>>>(hdr_in, hdr_rgb_out, channels, number_pixels);
}

void convertFloatToUint8(float* hdr, uint8_t* ldr, uint32_t number_values)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_values, block_size ); // Spawn enough blocks
    convertFloatToUint8Kernel<<<block_count, block_size>>>(hdr, ldr, number_values);
}
