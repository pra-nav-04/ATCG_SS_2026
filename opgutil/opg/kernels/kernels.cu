#include "opg/kernels/kernels.h"
#include "opg/kernels/launch.h"
#include "opg/exception.h"

#include "opg/hostdevice/color.h"
#include "opg/memory/devicebuffer.h"
#include "opg/memory/atomic.h"

#include <cooperative_groups.h>

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ


namespace opg {

__global__ void tonemap_srgb_kernel(uint32_t thread_count, BufferView<glm::vec3> input_hdr, BufferView<glm::u8vec3> output_ldr)
{
    uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index >= thread_count)
        return;
    output_ldr[thread_index] = make_srgb_color(input_hdr[thread_index]);
}

void tonemap_srgb(BufferView<glm::vec3> input_hdr, BufferView<glm::u8vec3> output_ldr)
{
    OPG_ASSERT(input_hdr.count == output_ldr.count);
    launch_linear_kernel(tonemap_srgb_kernel, input_hdr.count, input_hdr, output_ldr);
}

__global__ void tonemap_linear_kernel(uint32_t thread_count, BufferView<glm::vec3> input_hdr, BufferView<glm::u8vec3> output_ldr)
{
    uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index >= thread_count)
        return;
    output_ldr[thread_index] = static_cast<glm::u8vec3>(glm::clamp(input_hdr[thread_index], 0.0f, 1.0f)*255.0f);
}

void tonemap_linear(BufferView<glm::vec3> input_hdr, BufferView<glm::u8vec3> output_ldr)
{
    OPG_ASSERT(input_hdr.count == output_ldr.count);
    launch_linear_kernel(tonemap_linear_kernel, input_hdr.count, input_hdr, output_ldr);
}

template <typename T>
__forceinline__ __device__ T shfl_down(cooperative_groups::coalesced_group &group, T elem, uint32_t delta)
{
    return group.shfl_down(elem, delta);
}

template <int N, typename T, glm::qualifier Q>
__forceinline__ __device__ glm::vec<N, T, Q> shfl_down(cooperative_groups::coalesced_group &group, glm::vec<N, T, Q> elem, uint32_t delta)
{
    glm::vec<N, T, Q> result;
    for (int i = 0; i < N; ++i)
    {
        result[i] = group.shfl_down(elem[i], delta);
    }
    return result;
}


template <typename T>
__global__ void accumulate_samples_kernel(TensorView<T, 3> sample_buffer, TensorView<T, 2> accum_buffer, uint32_t existing_sample_count)
{
    glm::uvec3 thread_index = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);
    if (thread_index.x >= accum_buffer.counts[1] || thread_index.y >= accum_buffer.counts[0])
        return;

    //accum_buffer[thread_index] = value;

    uint32_t new_sample_count = sample_buffer.counts[0];
    uint32_t total_sample_count = existing_sample_count + new_sample_count;

    T acc = T(0);
    for (uint32_t i = threadIdx.z; i < new_sample_count; i += blockDim.z)
    {
        T value = sample_buffer[i][thread_index.y][thread_index.x];
        acc += value;
    }

    auto warp = cooperative_groups::coalesced_threads();

    acc += shfl_down(warp, acc, 16);
    acc += shfl_down(warp, acc, 8);
    acc += shfl_down(warp, acc, 4);
    acc += shfl_down(warp, acc, 2);
    acc += shfl_down(warp, acc, 1);

    if (threadIdx.z == 0)
    {
        float acc_factor = 1.0f / static_cast<float>(total_sample_count);
        float existing_factor = static_cast<float>(existing_sample_count) / static_cast<float>(total_sample_count);

        T existing_value = accum_buffer[thread_index.y][thread_index.x];

        accum_buffer[thread_index.y][thread_index.x] = acc_factor * acc + existing_factor * existing_value;
    }
}

template <typename T>
void accumulate_samples(TensorView<T, 3> sample_buffer, TensorView<T, 2> accum_buffer, uint32_t existing_sample_count)
{
    OPG_ASSERT(sample_buffer.counts[1] == accum_buffer.counts[0] && sample_buffer.counts[2] == accum_buffer.counts[1]);

    glm::uvec3 block_size = glm::uvec3(1, 1, 32);
    glm::uvec3 block_count = ceil_div<glm::uvec3>(glm::uvec3(accum_buffer.counts[1], accum_buffer.counts[0], 1), block_size);
    accumulate_samples_kernel<<<glm2cuda(block_count), glm2cuda(block_size)>>>(sample_buffer, accum_buffer, existing_sample_count);
}
template void accumulate_samples<float>(TensorView<float, 3> sample_buffer, TensorView<float, 2> accum_buffer, uint32_t existing_sample_count);
template void accumulate_samples<glm::vec1>(TensorView<glm::vec1, 3> sample_buffer, TensorView<glm::vec1, 2> accum_buffer, uint32_t existing_sample_count);
template void accumulate_samples<glm::vec2>(TensorView<glm::vec2, 3> sample_buffer, TensorView<glm::vec2, 2> accum_buffer, uint32_t existing_sample_count);
template void accumulate_samples<glm::vec3>(TensorView<glm::vec3, 3> sample_buffer, TensorView<glm::vec3, 2> accum_buffer, uint32_t existing_sample_count);
template void accumulate_samples<glm::vec4>(TensorView<glm::vec4, 3> sample_buffer, TensorView<glm::vec4, 2> accum_buffer, uint32_t existing_sample_count);


__global__ void cumsum_kernel_32(TensorView<float, 2> data)
{
    glm::uvec3 thread_index = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);
    if (thread_index.x >= data.counts[0] || thread_index.y >= 32)
        return;

    uint32_t lane_index = thread_index.x;
    uint32_t local_thread_index = thread_index.y;

    auto warp = cooperative_groups::coalesced_threads();

    float acc = 0;
    for (uint32_t i = local_thread_index; i < data.counts[1]; i += 32)
    {
        float value = data[lane_index][i];

#define WARP_SHFL_ADD(x) do { \
                float temp = warp.shfl(value, local_thread_index - (local_thread_index % (x)) + (x)/2 - 1); \
                if ((local_thread_index % (x)) >= ((x)/2)) value += temp; \
            } while (false)
        WARP_SHFL_ADD(2);
        WARP_SHFL_ADD(4);
        WARP_SHFL_ADD(8);
        WARP_SHFL_ADD(16);
        WARP_SHFL_ADD(32);
#undef WARP_SHFL_ADD

        acc += value;
        data[lane_index][i] = acc;

        // Take accumulator for last lane!
        acc = warp.shfl(acc, 31);
    }
}


template <typename T>
__global__ void copy_kernel(glm::uvec2 thread_count, TensorView<T, 2> input_view, TensorView<T, 2> output_view)
{
    glm::uvec3 thread_index = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);
    if (thread_index.x >= thread_count.x || thread_index.y >= thread_count.y)
        return;

    for (uint32_t y = thread_index.y; y < output_view.counts[0]; y += blockDim.y * gridDim.y)
    {
        for (uint32_t x = thread_index.x; x < output_view.counts[1]; x += blockDim.x * gridDim.x)
        {
            glm::uvec2 coord = glm::vec2(x, y);
            output_view(coord).value() = input_view(coord).value();
        }
    }
}

template <typename T>
void copy(TensorView<T, 2> input_view, TensorView<T, 2> output_view)
{
    OPG_ASSERT(input_view.isValid() && output_view.isValid());
    OPG_ASSERT(input_view.counts[0] == output_view.counts[0]);
    OPG_ASSERT(input_view.counts[1] == output_view.counts[1]);
    launch_2d_kernel(copy_kernel<T>, glm::uvec2(output_view.counts[1], output_view.counts[0]), input_view, output_view);
}
template void copy<float>(TensorView<float, 2> input_view, TensorView<float, 2> output_view);
template void copy<glm::vec1>(TensorView<glm::vec1, 2> input_view, TensorView<glm::vec1, 2> output_view);
template void copy<glm::vec2>(TensorView<glm::vec2, 2> input_view, TensorView<glm::vec2, 2> output_view);
template void copy<glm::vec3>(TensorView<glm::vec3, 2> input_view, TensorView<glm::vec3, 2> output_view);
template void copy<glm::vec4>(TensorView<glm::vec4, 2> input_view, TensorView<glm::vec4, 2> output_view);

} // namespace opg
