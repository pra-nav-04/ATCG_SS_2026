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

void scaleRgb(glm::vec3* rgb, float scale, uint32_t number_pixels);
void gammaCorrectRgb(glm::vec3* rgb, float gamma, uint32_t number_pixels);
void histogramBins(glm::vec3* hsv_values, uint32_t* bins, const float* min_value, const float* max_value, uint32_t number_pixels, uint32_t number_bins);
void mapHistogramBrightness(glm::vec3* hsv_values, const float* accum_probs, const float* min_value, const float* max_value, uint32_t number_pixels, uint32_t number_bins);
