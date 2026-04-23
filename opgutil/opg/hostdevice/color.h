#pragma once

#include <cstdint>

#include "opg/preprocessor.h"
#include "opg/glmwrapper.h"

//
// Helper functions for sRGB linear<->gamma color conversion
// See https://en.wikipedia.org/wiki/SRGB
//

template <typename T>
OPG_INLINE OPG_HOSTDEVICE T apply_srgb_gamma(const T &linear_color)
{
    // Proper sRGB curve...
    auto cond = glm::lessThan(linear_color, T(0.0031308f));
    auto if_true = 12.92f * linear_color;
    auto if_false = 1.055f * glm::pow(linear_color, T(1.0f/2.4f)) - 0.055f;
    return glm::lerp(if_false, if_true, T(cond));
    // return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
}

template <typename T>
OPG_INLINE OPG_HOSTDEVICE T apply_inverse_srgb_gamma(const T &gamma_color)
{
    auto cond = glm::lessThan(gamma_color, T(0.04045f));
    auto if_true = gamma_color / T(12.92f);
    auto if_false = glm::pow((gamma_color + T(0.055)) / T(1.055), T(2.4f));
    return glm::lerp(if_false, if_true, T(cond));
}

OPG_INLINE OPG_HOSTDEVICE glm::u8vec3 make_srgb_color(const glm::vec3& linear_color)
{
    return static_cast<glm::u8vec3>(glm::clamp(apply_srgb_gamma(linear_color), 0.0f, 1.0f)*255.0f);
}


//
// rgb color to scalar weight functions
//

OPG_INLINE OPG_HOSTDEVICE float rgb_to_scalar_weight_max(const glm::vec3 &rgb)
{
    return glm::max(glm::max(rgb.x, rgb.y), rgb.z);
}

OPG_INLINE OPG_HOSTDEVICE float rgb_to_scalar_weight_mean(const glm::vec3 &rgb)
{
    return (rgb.x + rgb.y + rgb.z) / 3;
}

OPG_INLINE OPG_HOSTDEVICE float rgb_to_scalar_weight_luminance(const glm::vec3 &rgb)
{
    glm::vec3 luminance(0.21263901f, 0.71516868f, 0.07219232f);
    return glm::dot(luminance, rgb);
}

OPG_INLINE OPG_HOSTDEVICE float rgb_to_scalar_weight(const glm::vec3 &rgb)
{
    return rgb_to_scalar_weight_max(rgb);
}
