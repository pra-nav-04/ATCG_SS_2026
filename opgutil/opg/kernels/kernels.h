#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"
#include "opg/opgapi.h"

namespace opg {

// Apply simple sRGB tonemapping to floating-point HDR data and store it as 8-bit quantized LDR data.
OPG_API void tonemap_srgb(BufferView<glm::vec3> input_hdr, BufferView<glm::u8vec3> output_ldr);
// Apply simple linear tonemapping to floating-point HDR data and store it as 8-bit quantized LDR data.
OPG_API void tonemap_linear(BufferView<glm::vec3> input_hdr, BufferView<glm::u8vec3> output_ldr);

// Add an additional sample to the accumulation buffer which already contains the mean over a certain number of samples.
// The result stored in accum_buffer contains the sum over all samples, divided by the total sample count.
// The sample_buffer has shape (sample_count, height, width), and the accum_buffer has shape (height, width).
template <typename T>
OPG_API void accumulate_samples(TensorView<T, 3> sample_buffer, TensorView<T, 2> accum_buffer, uint32_t existing_sample_count = 0);

// Copy the contents from one 2D tensor view to another, irrespective of memory layout...
template <typename T>
OPG_API void copy(TensorView<T, 2> input_view, TensorView<T, 2> output_view);

} // namespace opg
