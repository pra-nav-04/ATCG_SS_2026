#pragma once

#include <string>

#include "opg/glmwrapper.h"

namespace opg {
    struct ImageData;
}

opg::ImageData tonemapImage(const opg::ImageData& img, const std::string& mode);

void extractRgbFromMultiChannel(float* hdr_in, glm::vec3* hdr_rgb_out, uint32_t channels, uint32_t number_pixels);
void convertFloatToUint8(float* hdr, uint8_t* ldr, uint32_t number_values);

void maxValue(float* values, float* max_value, uint32_t number_values);
void minValue(float* values, float* max_value, uint32_t number_values);
void brightnessMinMax(glm::vec3* hsv_values, float* min_value, float* max_value, uint32_t number_pixels);
void boxFilterRgb(glm::vec3 *hdr_rgb_in, glm::vec3 *hdr_rgb_out, uint32_t filter_size, uint32_t width, uint32_t height);

void convertRgbToHsvBrightness(glm::vec3* rgb, glm::vec3* hsv, uint32_t number_pixels);
void convertHsvToRgb(glm::vec3* hsv, glm::vec3* rgb, uint32_t number_pixels);

// TODO: declare your functions here
//