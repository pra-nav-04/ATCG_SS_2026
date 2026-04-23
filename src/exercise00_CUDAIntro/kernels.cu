#include <cuda_runtime.h>
#include "opg/hostdevice/random.h"
#include "opg/glmwrapper.h"
#include "opg/hostdevice/misc.h"
#include <cstdint>
#include <cmath>

#include "kernels.h"

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ

namespace {

constexpr int ARRAY_BLOCK_SIZE = 256;
constexpr int IMAGE_BLOCK_SIZE = 16;
constexpr int GEMM_TILE_SIZE   = 16;

__device__ __forceinline__ int clampCoord(int value, int upper_bound)
{
    return max(0, min(value, upper_bound - 1));
}

__device__ __forceinline__ int pixelIndex(int x, int y, int channel, int width, int channels)
{
    return (y * width + x) * channels + channel;
}

__global__ void generate_sequence_kernel(int* output, int count)
{
    int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }

    output[index] = index + 1;
}

__global__ void multiply_int_array_kernel(int* data, int count, int multiplier)
{
    int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }

    data[index] *= multiplier;
}

__global__ void sobel_derivative_x_pass1_kernel(const uint8_t* input, float* temp, int width, int height, int channels)
{
    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
    {
        return;
    }

    int left_x  = clampCoord(x - 1, width);
    int right_x = clampCoord(x + 1, width);

    for (int channel = 0; channel < channels; ++channel)
    {
        float left  = static_cast<float>(input[pixelIndex(left_x, y, channel, width, channels)]);
        float right = static_cast<float>(input[pixelIndex(right_x, y, channel, width, channels)]);
        temp[pixelIndex(x, y, channel, width, channels)] = right - left;
    }
}

__global__ void sobel_derivative_x_pass2_kernel(const float* temp, float* output, int width, int height, int channels)
{
    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
    {
        return;
    }

    int top_y    = clampCoord(y - 1, height);
    int bottom_y = clampCoord(y + 1, height);

    for (int channel = 0; channel < channels; ++channel)
    {
        float top    = temp[pixelIndex(x, top_y, channel, width, channels)];
        float center = temp[pixelIndex(x, y, channel, width, channels)];
        float bottom = temp[pixelIndex(x, bottom_y, channel, width, channels)];
        output[pixelIndex(x, y, channel, width, channels)] = top + 2.0f * center + bottom;
    }
}

__global__ void sobel_derivative_y_pass1_kernel(const uint8_t* input, float* temp, int width, int height, int channels)
{
    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
    {
        return;
    }

    int left_x  = clampCoord(x - 1, width);
    int right_x = clampCoord(x + 1, width);

    for (int channel = 0; channel < channels; ++channel)
    {
        float left   = static_cast<float>(input[pixelIndex(left_x, y, channel, width, channels)]);
        float center = static_cast<float>(input[pixelIndex(x, y, channel, width, channels)]);
        float right  = static_cast<float>(input[pixelIndex(right_x, y, channel, width, channels)]);
        temp[pixelIndex(x, y, channel, width, channels)] = left + 2.0f * center + right;
    }
}

__global__ void sobel_derivative_y_pass2_kernel(const float* temp, float* output, int width, int height, int channels)
{
    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
    {
        return;
    }

    int top_y    = clampCoord(y - 1, height);
    int bottom_y = clampCoord(y + 1, height);

    for (int channel = 0; channel < channels; ++channel)
    {
        float top    = temp[pixelIndex(x, top_y, channel, width, channels)];
        float bottom = temp[pixelIndex(x, bottom_y, channel, width, channels)];
        output[pixelIndex(x, y, channel, width, channels)] = bottom - top;
    }
}

__global__ void sobel_magnitude_kernel(const uint8_t* input, const float* grad_x, const float* grad_y, uint8_t* output, int width, int height, int channels)
{
    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
    {
        return;
    }

    for (int channel = 0; channel < channels; ++channel)
    {
        int index = pixelIndex(x, y, channel, width, channels);
        if (channels == 4 && channel == 3)
        {
            output[index] = input[index];
            continue;
        }

        float magnitude = sqrtf(grad_x[index] * grad_x[index] + grad_y[index] * grad_y[index]);
        output[index] = static_cast<uint8_t>(fminf(magnitude, 255.0f));
    }
}

__global__ void generate_random_and_count_kernel(float* random_values, int total_count, float threshold, int* counter)
{
    int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_count)
    {
        return;
    }

    const uint32_t seed_index = static_cast<uint32_t>(index);
    PCG32 rng(sampleTEA64(seed_index, 0u), sampleTEA64(seed_index, 1u));
    float value = rng.nextFloat();
    random_values[index] = value;

    if (value > threshold)
    {
        atomicAdd(counter, 1);
    }
}

__global__ void matrix_multiply_kernel(const float* lhs, const float* rhs, float* output, int lhs_rows, int lhs_cols, int rhs_cols)
{
    __shared__ float lhs_tile[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    __shared__ float rhs_tile[GEMM_TILE_SIZE][GEMM_TILE_SIZE];

    const int local_x = static_cast<int>(threadIdx.x);
    const int local_y = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    float accum = 0.0f;

    for (int tile_start = 0; tile_start < lhs_cols; tile_start += GEMM_TILE_SIZE)
    {
        const int lhs_col = tile_start + local_x;
        const int rhs_row = tile_start + local_y;

        lhs_tile[local_y][local_x] = (row < lhs_rows && lhs_col < lhs_cols)
            ? lhs[row * lhs_cols + lhs_col]
            : 0.0f;
        rhs_tile[local_y][local_x] = (rhs_row < lhs_cols && col < rhs_cols)
            ? rhs[rhs_row * rhs_cols + col]
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE_SIZE; ++k)
        {
            accum += lhs_tile[local_y][k] * rhs_tile[k][local_x];
        }

        __syncthreads();
    }

    if (row < lhs_rows && col < rhs_cols)
    {
        output[row * rhs_cols + col] = accum;
    }
}

} // namespace

void launchGenerateSequence(int* d_output, int count)
{
    dim3 block_size(ARRAY_BLOCK_SIZE);
    dim3 block_count(ceil_div(count, ARRAY_BLOCK_SIZE));
    generate_sequence_kernel<<<block_count, block_size>>>(d_output, count);
}

void launchMultiplyIntArray(int* d_data, int count, int multiplier)
{
    dim3 block_size(ARRAY_BLOCK_SIZE);
    dim3 block_count(ceil_div(count, ARRAY_BLOCK_SIZE));
    multiply_int_array_kernel<<<block_count, block_size>>>(d_data, count, multiplier);
}

void launchSobelDerivativeXPass1(const uint8_t* d_input, float* d_temp, int width, int height, int channels)
{
    dim3 block_size(IMAGE_BLOCK_SIZE, IMAGE_BLOCK_SIZE);
    dim3 block_count(ceil_div(width, IMAGE_BLOCK_SIZE), ceil_div(height, IMAGE_BLOCK_SIZE));
    sobel_derivative_x_pass1_kernel<<<block_count, block_size>>>(d_input, d_temp, width, height, channels);
}

void launchSobelDerivativeXPass2(const float* d_temp, float* d_output, int width, int height, int channels)
{
    dim3 block_size(IMAGE_BLOCK_SIZE, IMAGE_BLOCK_SIZE);
    dim3 block_count(ceil_div(width, IMAGE_BLOCK_SIZE), ceil_div(height, IMAGE_BLOCK_SIZE));
    sobel_derivative_x_pass2_kernel<<<block_count, block_size>>>(d_temp, d_output, width, height, channels);
}

void launchSobelDerivativeYPass1(const uint8_t* d_input, float* d_temp, int width, int height, int channels)
{
    dim3 block_size(IMAGE_BLOCK_SIZE, IMAGE_BLOCK_SIZE);
    dim3 block_count(ceil_div(width, IMAGE_BLOCK_SIZE), ceil_div(height, IMAGE_BLOCK_SIZE));
    sobel_derivative_y_pass1_kernel<<<block_count, block_size>>>(d_input, d_temp, width, height, channels);
}

void launchSobelDerivativeYPass2(const float* d_temp, float* d_output, int width, int height, int channels)
{
    dim3 block_size(IMAGE_BLOCK_SIZE, IMAGE_BLOCK_SIZE);
    dim3 block_count(ceil_div(width, IMAGE_BLOCK_SIZE), ceil_div(height, IMAGE_BLOCK_SIZE));
    sobel_derivative_y_pass2_kernel<<<block_count, block_size>>>(d_temp, d_output, width, height, channels);
}

void launchSobelMagnitude(const uint8_t* d_input, const float* d_grad_x, const float* d_grad_y, uint8_t* d_output, int width, int height, int channels)
{
    dim3 block_size(IMAGE_BLOCK_SIZE, IMAGE_BLOCK_SIZE);
    dim3 block_count(ceil_div(width, IMAGE_BLOCK_SIZE), ceil_div(height, IMAGE_BLOCK_SIZE));
    sobel_magnitude_kernel<<<block_count, block_size>>>(d_input, d_grad_x, d_grad_y, d_output, width, height, channels);
}

void launchGenerateRandomAndCount(float* d_random_values, int total_count, float threshold, int* d_counter)
{
    dim3 block_size(ARRAY_BLOCK_SIZE);
    dim3 block_count(ceil_div(total_count, ARRAY_BLOCK_SIZE));
    generate_random_and_count_kernel<<<block_count, block_size>>>(d_random_values, total_count, threshold, d_counter);
}

void launchMatrixMultiply(const float* d_lhs, const float* d_rhs, float* d_output, int lhs_rows, int lhs_cols, int rhs_cols)
{
    dim3 block_size(GEMM_TILE_SIZE, GEMM_TILE_SIZE);
    dim3 block_count(ceil_div(rhs_cols, GEMM_TILE_SIZE), ceil_div(lhs_rows, GEMM_TILE_SIZE));
    matrix_multiply_kernel<<<block_count, block_size>>>(d_lhs, d_rhs, d_output, lhs_rows, lhs_cols, rhs_cols);
}
