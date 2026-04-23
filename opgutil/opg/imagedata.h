#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "opg/opgapi.h"

namespace opg {

enum ImageFormat : int
{
    FORMAT_R_UINT8,
    FORMAT_RG_UINT8,
    FORMAT_RGB_UINT8,
    FORMAT_RGBA_UINT8,

    FORMAT_RGB_FLOAT,
    FORMAT_RGBA_FLOAT,
};

struct ImageData
{
    std::vector<unsigned char> data;
    uint32_t width;
    uint32_t height;
    ImageFormat format;
};

OPG_API uint32_t getImageFormatChannelCount(ImageFormat format);
OPG_API uint32_t getImageFormatChannelSize(ImageFormat format);
OPG_API uint32_t getImageFormatPixelSize(ImageFormat format);

OPG_API bool readImage(const char *filename, ImageData &image);

OPG_API void writeImagePNG(const char *filename, const ImageData &image);
OPG_API void writeImageEXR(const char *filename, const ImageData &image, bool halfPrecision=true);

} // end namespace opg
