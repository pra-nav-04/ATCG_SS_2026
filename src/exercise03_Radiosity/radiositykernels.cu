#include "radiositykernels.h"

#include "opg/hostdevice/misc.h"

#pragma cuda_source_property_format=OBJ

namespace {

__global__ void initializeRadiosityBufferKernel(
    opg::BufferView<glm::vec3> radiosity,
    opg::BufferView<glm::vec3> emissions)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= radiosity.count)
        return;

    radiosity[idx] = emissions[idx];
}

__global__ void jacobiRadiosityStepKernel(
    opg::BufferView<float> form_factor_matrix,
    uint32_t matrix_size,
    opg::BufferView<glm::vec3> albedos,
    opg::BufferView<glm::vec3> emissions,
    opg::BufferView<glm::vec3> previous_radiosity,
    opg::BufferView<glm::vec3> next_radiosity,
    opg::BufferView<float> deltas,
    float lambda)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= matrix_size)
        return;

    glm::vec3 transport_sum(0.0f);
    const uint32_t row_offset = row * matrix_size;
    for (uint32_t col = 0; col < matrix_size; ++col)
    {
        transport_sum += form_factor_matrix[row_offset + col] * previous_radiosity[col];
    }

    const glm::vec3 previous = previous_radiosity[row];
    const glm::vec3 residual = emissions[row] - (previous - albedos[row] * transport_sum);
    const glm::vec3 updated = previous + lambda * residual;

    next_radiosity[row] = updated;

    const glm::vec3 diff = glm::abs(updated - previous);
    deltas[row] = glm::max(diff.x, glm::max(diff.y, diff.z));
}

} // namespace

void initializeRadiosityBuffer(
    opg::BufferView<glm::vec3> radiosity,
    opg::BufferView<glm::vec3> emissions)
{
    const uint32_t block_size = 256;
    const uint32_t grid_size = opg::divideRoundUp(radiosity.count, block_size);
    initializeRadiosityBufferKernel<<<grid_size, block_size>>>(radiosity, emissions);
    CUDA_SYNC_CHECK();
}

void jacobiRadiosityStep(
    opg::BufferView<float> form_factor_matrix,
    uint32_t matrix_size,
    opg::BufferView<glm::vec3> albedos,
    opg::BufferView<glm::vec3> emissions,
    opg::BufferView<glm::vec3> previous_radiosity,
    opg::BufferView<glm::vec3> next_radiosity,
    opg::BufferView<float> deltas,
    float lambda)
{
    const uint32_t block_size = 256;
    const uint32_t grid_size = opg::divideRoundUp(matrix_size, block_size);
    jacobiRadiosityStepKernel<<<grid_size, block_size>>>(
        form_factor_matrix,
        matrix_size,
        albedos,
        emissions,
        previous_radiosity,
        next_radiosity,
        deltas,
        lambda);
    CUDA_SYNC_CHECK();
}
