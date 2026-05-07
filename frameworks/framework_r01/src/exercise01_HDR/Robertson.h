#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

#include "opg/glmwrapper.h"

// forward declaration
namespace opg {
    struct ImageData;
}

void splitChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels);
void splitChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels);
void mergeChannels(glm::u8vec3* pixels, uint8_t* red, uint8_t* green, uint8_t* blue, int number_pixels);
void mergeChannels(glm::f32vec3* pixels, float* red, float* green, float* blue, int number_pixels);

void calcMask(uint8_t* values, bool* underexposed_mask, uint32_t number_values, uint32_t values_per_image);
void countValues(uint8_t* values, bool* underexposed_mask, uint32_t* counters, uint32_t number_values, uint32_t values_per_image);
void calcWeights(uint8_t* values, float* weights, uint32_t number_values);

void initInvCrf(float* I, uint32_t number_values);
void normInvCrf(float* I, uint32_t number_values);

// TODO: declare your functions here
//

std::pair<opg::ImageData, std::vector<std::vector<float>>> robertson(const std::vector<opg::ImageData> &imgs, const std::vector<float> &exposures, size_t max_iterations = 10);
