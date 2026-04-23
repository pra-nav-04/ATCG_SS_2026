#pragma once

#include <cstdint>
#include "opg/hostdevice/misc.h"


namespace opg {

constexpr uint32_t MaxThreadsPerBlock = 512;

template <typename K, typename... Args>
void launch_single_kernel(K kernel, Args... args)
{
    kernel<<<1, 1>>>(std::forward<Args>(args)...);
}


template <typename K, typename... Args>
void launch_linear_kernel(K kernel, uint32_t thread_count, Args... args)
{
    uint32_t block_size = MaxThreadsPerBlock;
    uint32_t block_count = ceil_div(thread_count, block_size);
    kernel<<<block_count, block_size>>>(thread_count, std::forward<Args>(args)...);
}


template <typename K, typename... Args>
void launch_1d_kernel(K kernel, glm::uvec1 thread_count, Args... args)
{
    uint32_t block_size = MaxThreadsPerBlock;
    uint32_t block_count = ceil_div(thread_count.x, block_size);
    kernel<<<block_count, block_size>>>(thread_count, std::forward<Args>(args)...);
}

template <typename K, typename... Args>
void launch_2d_kernel(K kernel, glm::uvec2 thread_count, Args... args)
{
    glm::uvec3 block_size = glm::uvec3(16, 16, 1);
    glm::uvec3 block_count = ceil_div(glm::uvec3(thread_count.x, thread_count.y, 1), block_size);
    kernel<<<glm2cuda(block_count), glm2cuda(block_size)>>>(thread_count, std::forward<Args>(args)...);
}

template <typename K, typename... Args>
void launch_3d_kernel(K kernel, glm::uvec3 thread_count, Args... args)
{
    glm::uvec3 block_size = glm::uvec3(8, 8, 8);
    glm::uvec3 block_count = ceil_div(thread_count, block_size);
    kernel<<<glm2cuda(block_count), glm2cuda(block_size)>>>(thread_count, std::forward<Args>(args)...);
}

} // namespace opg
