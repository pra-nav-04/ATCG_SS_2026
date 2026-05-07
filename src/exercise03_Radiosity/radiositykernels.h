#pragma once

#include "opg/glmwrapper.h"

#include "opg/memory/bufferview.h"

void initializeRadiosityBuffer(
    opg::BufferView<glm::vec3> radiosity,
    opg::BufferView<glm::vec3> emissions);

void jacobiRadiosityStep(
    opg::BufferView<float> form_factor_matrix,
    uint32_t matrix_size,
    opg::BufferView<glm::vec3> albedos,
    opg::BufferView<glm::vec3> emissions,
    opg::BufferView<glm::vec3> previous_radiosity,
    opg::BufferView<glm::vec3> next_radiosity,
    opg::BufferView<float> deltas,
    float lambda);
