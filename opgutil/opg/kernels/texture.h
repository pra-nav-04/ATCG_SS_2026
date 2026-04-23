#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"
#include "opg/opgapi.h"

namespace opg {

// Sample the contents from a cuda texture object into a normal floating-point tensor.
template <typename T = glm::vec4, int Dim=2>
OPG_API void texture_to_float(cudaTextureObject_t tex, TensorView<T, Dim> output);

// Compute the maximum value of a cuda texture object.
template <typename T = glm::vec4>
OPG_API void texture_max_reduce_3d(cudaTextureObject_t tex, const glm::uvec3 &tex_size, T *result);

} // namespace opg
