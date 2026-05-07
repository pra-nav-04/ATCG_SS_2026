#include "Robertson.h"

#include "opg/hostdevice/misc.h"

#include <cuda_runtime.h>
#include <algorithm>

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ

template <class Vec3T>
__global__ void splitChannelsKernel(Vec3T* pixels, typename Vec3T::value_type* red, typename Vec3T::value_type* green, typename Vec3T::value_type* blue, int number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    const uint32_t img_index = gid;
    red[gid] = pixels[img_index].x;
    green[gid] = pixels[img_index].y;
    blue[gid] = pixels[img_index].z;
}

template <class Vec3T>
__global__ void mergeChannelsKernel(Vec3T* pixels, typename Vec3T::value_type* red, typename Vec3T::value_type* green, typename Vec3T::value_type* blue, int number_pixels)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_pixels)
    {
        return;
    }

    const uint32_t img_index = gid;
    pixels[img_index].x = red[gid];
    pixels[img_index].y = green[gid];
    pixels[img_index].z = blue[gid];
}

__global__ void calcMaskKernel(uint8_t* values, bool* underexposed_mask, uint32_t number_values, uint32_t values_per_image)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    uint32_t number_imgs = number_values / values_per_image;
    float mean = 0.0f;
    for (size_t i = 0; i < number_imgs; i++)
    {
        mean += float(values[gid + i * values_per_image]) / float(number_imgs);
    }

    // Mask out under- *and* overexposed pixels.
    underexposed_mask[gid] = (mean < 5.0f) || (mean > 250.0f);
}

__global__ void countValuesKernel(uint8_t* values, bool* underexposed_mask, uint32_t* counters, uint32_t number_values, uint32_t values_per_image)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }
    
    size_t j = gid % values_per_image;
    if (underexposed_mask[j])
    {
        return;
    }

    atomicAdd(counters + values[gid], 1);
}

__global__ void calcWeightsKernel(uint8_t* values, float* weights, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    float nom = float(values[gid]) - 127.5f;
    float denom = 127.5f * 127.5f;
    float w = fmaxf(expf(-4.0f * nom * nom / denom) - expf(-4.0f), 0.0f);
    weights[gid] = w;
}

__global__ void normInvCrfKernel(float* I, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    float ref = I[(number_values - 1) / 2];
    I[gid] /= ref;
}

__global__ void initInvCrfKernel(float* I, uint32_t number_values)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= number_values)
    {
        return;
    }

    I[gid] = float(gid);
}

// TODO: put your CUDA kernels and the host functions which launch the kernels here
//

void splitChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    splitChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);
}

void splitChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    splitChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);
}

void mergeChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>( number_pixels, block_size ); // Spawn enough blocks
    mergeChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);
}

void mergeChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(number_pixels, block_size); // Spawn enough blocks
    mergeChannelsKernel<<<block_count, block_size>>>(pixels, red, green, blue, number_pixels);
}

void calcMask(uint8_t* values, bool* underexposed_mask, uint32_t number_values, uint32_t values_per_image)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(values_per_image, block_size); // Spawn enough blocks
    calcMaskKernel<<<block_count, block_size>>>(values, underexposed_mask, number_values, values_per_image);
}

void countValues(uint8_t* values, bool* underexposed_mask, uint32_t* counters, uint32_t number_values, uint32_t values_per_image)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
    countValuesKernel<<<block_count, block_size>>>(values, underexposed_mask, counters, number_values, values_per_image);
}

void calcWeights(uint8_t* values, float* weights, uint32_t number_values)
{
    // launch kernel
    const int block_size  = 512; // 512 is a size that works well with modern GPUs.
    const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
    calcWeightsKernel<<<block_count, block_size>>>(values, weights, number_values);
}

void initInvCrf(float* I, uint32_t number_values)
{
    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
        initInvCrfKernel<<<block_count, block_size>>>(I, number_values);
    }
    cudaDeviceSynchronize();

    normInvCrf(I, number_values);
}

void normInvCrf(float* I, uint32_t number_values)
{
    // launch kernel
    {
        const int block_size  = 512; // 512 is a size that works well with modern GPUs.
        const int block_count = ceil_div<int>(number_values, block_size); // Spawn enough blocks
        normInvCrfKernel<<<block_count, block_size>>>(I, number_values);
    }
}
