#include "opg/kernels/texture.h"
#include "opg/kernels/launch.h"
#include "opg/exception.h"

#include "opg/memory/devicebuffer.h"
#include "opg/memory/atomic.h"

#include <cooperative_groups.h>

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format=OBJ


namespace opg {

template <typename T, int Dim>
__global__ void texture_to_float_kernel(cudaTextureObject_t tex, TensorView<T, Dim> output_buffer)
{
    static_assert(Dim >= 1 && Dim <= 3, "Only 1 to 3 dimensional texture objects are supported!");
    glm::uvec3 thread_index = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);
    glm::uvec3 texture_size = glm::uvec3(
        Dim >= 1 ? output_buffer.counts[Dim-1] : 1,
        Dim >= 2 ? output_buffer.counts[Dim-2] : 1,
        Dim >= 3 ? output_buffer.counts[Dim-3] : 1
    );
    if (glm::any(glm::greaterThanEqual(thread_index, texture_size)))
        return;

    glm::vec<Dim, uint32_t> pixel_index_u = glm::vec<Dim, uint32_t>(thread_index);
    glm::vec<Dim, float> pixel_index_f = glm::vec<Dim, float>(thread_index);
    glm::vec<Dim, float> tex_size_f = glm::vec<Dim, float>(texture_size);
    glm::vec<Dim, float> coord_f = (pixel_index_f + 0.5f) / tex_size_f;

    T value;
    if constexpr (Dim == 1)
        value = tex1D<T>(tex, coord_f);
    else if constexpr (Dim == 2)
        value = tex2D<T>(tex, coord_f);
    else if constexpr (Dim == 3)
        value = tex3D<T>(tex, coord_f);

    output_buffer(pixel_index_u) = value;
}

template <typename T, int Dim>
void texture_to_float(cudaTextureObject_t tex, TensorView<T, Dim> output_buffer)
{
    static_assert(Dim >= 1 && Dim <= 3, "Only 1 to 3 dimensional texture objects are supported!");
    glm::uvec3 thread_count;
    glm::uvec3 block_size;
    if constexpr (Dim == 1)
    {
        thread_count = glm::uvec3(output_buffer.counts[0], 1, 1);
        block_size = glm::uvec3(512, 1, 1);
    }
    else if constexpr (Dim == 2)
    {
        thread_count = glm::uvec3(output_buffer.counts[1], output_buffer.counts[0], 1);
        block_size = glm::uvec3(16, 16, 1);
    }
    else if constexpr (Dim == 3)
    {
        thread_count = glm::uvec3(output_buffer.counts[2], output_buffer.counts[1], output_buffer.counts[0]);
        block_size = glm::uvec3(8, 8, 8);
    }
    glm::uvec3 block_count = ceil_div<glm::uvec3>(thread_count, block_size);
    texture_to_float_kernel<<<glm2cuda(block_count), glm2cuda(block_size)>>>(tex, output_buffer);
}
#define INSTANTIATE_Type_Dim_texture_to_float(Type, Dim) \
    template void texture_to_float<Type, Dim>(cudaTextureObject_t tex, TensorView<Type, Dim> output_buffer);
#define INSTANTIATE_Dim_texture_to_float(Dim) \
    INSTANTIATE_Type_Dim_texture_to_float(float, Dim) \
    INSTANTIATE_Type_Dim_texture_to_float(glm::vec1, Dim) \
    INSTANTIATE_Type_Dim_texture_to_float(glm::vec2, Dim) \
    INSTANTIATE_Type_Dim_texture_to_float(glm::vec3, Dim) \
    INSTANTIATE_Type_Dim_texture_to_float(glm::vec4, Dim)
INSTANTIATE_Dim_texture_to_float(1)
INSTANTIATE_Dim_texture_to_float(2)
INSTANTIATE_Dim_texture_to_float(3)
#undef INSTANTIATE_Dim_texture_to_float
#undef INSTANTIATE_Type_Dim_texture_to_float

template <typename T>
__global__ void texture_max_reduce_3d_kernel(cudaTextureObject_t tex, glm::uvec3 tex_size, T *output)
{
    glm::uvec3 thread_count = cuda2glm(blockDim) * cuda2glm(gridDim);
    glm::uvec3 thread_index = cuda2glm(threadIdx) + cuda2glm(blockIdx) * cuda2glm(blockDim);

    T max_value = std::numeric_limits<T>::min(); // not negative infinity, just in case this is called with integers....
    for (uint32_t z = thread_index.z; z < tex_size.z; z += thread_count.z)
    {
        for (uint32_t y = thread_index.y; y < tex_size.y; y += thread_count.y)
        {
            for (uint32_t x = thread_index.x; x < tex_size.x; x += thread_count.x)
            {
                glm::vec3 pixel_index = glm::vec3(x, y, z);
                glm::vec3 uvw = (pixel_index + 0.5f) / glm::vec3(tex_size);

                T value = tex3D<T>(tex, uvw);
                max_value = glm::max(value, max_value);
            }
        }
    }

    atomicMax(output, max_value);
}

template <typename T>
void texture_max_reduce_3d(cudaTextureObject_t tex, const glm::uvec3 &tex_size, T *result)
{
    // TODO optimize block size and count?
    glm::uvec3 block_size = glm::uvec3(4, 4, 4);
    glm::uvec3 block_count = glm::uvec3(4, 4, 4); // ceil_div<glm::uvec3>(tex_size, block_size);

    T temp_value = std::numeric_limits<T>::min();
    opg::DeviceBuffer<T> temp(1);
    temp.upload(&temp_value);
    texture_max_reduce_3d_kernel<<<glm2cuda(block_count), glm2cuda(block_size)>>>(tex, tex_size, temp.data());
    temp.download(result);
}
template OPG_API void texture_max_reduce_3d<float>(cudaTextureObject_t tex, const glm::uvec3 &tex_size, float *result);
template OPG_API void texture_max_reduce_3d<glm::vec1>(cudaTextureObject_t tex, const glm::uvec3 &tex_size, glm::vec1 *result);
template OPG_API void texture_max_reduce_3d<glm::vec2>(cudaTextureObject_t tex, const glm::uvec3 &tex_size, glm::vec2 *result);
template OPG_API void texture_max_reduce_3d<glm::vec3>(cudaTextureObject_t tex, const glm::uvec3 &tex_size, glm::vec3 *result);
template OPG_API void texture_max_reduce_3d<glm::vec4>(cudaTextureObject_t tex, const glm::uvec3 &tex_size, glm::vec4 *result);

} // namespace opg
