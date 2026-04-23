#include "opg/imagedata.h"

#include "opg/exception.h"

#include <tinyexr/tinyexr.h>
#include <stb_image.h>
#include <stb_image_write.h>

namespace opg {

uint32_t getImageFormatChannelCount(ImageFormat format)
{
    switch (format)
    {
        case ImageFormat::FORMAT_R_UINT8:
            return 1;
        case ImageFormat::FORMAT_RG_UINT8:
            return 2;
        case ImageFormat::FORMAT_RGB_UINT8:
        case ImageFormat::FORMAT_RGB_FLOAT:
            return 3;
        case ImageFormat::FORMAT_RGBA_UINT8:
        case ImageFormat::FORMAT_RGBA_FLOAT:
            return 4;
        default:
            throw std::runtime_error("getImageFormatChannelCount: unrecognized image format");
    }
}

uint32_t getImageFormatChannelSize(ImageFormat format)
{
    switch (format)
    {
        case ImageFormat::FORMAT_R_UINT8:
        case ImageFormat::FORMAT_RG_UINT8:
        case ImageFormat::FORMAT_RGB_UINT8:
        case ImageFormat::FORMAT_RGBA_UINT8:
            return sizeof(uint8_t);
        case ImageFormat::FORMAT_RGB_FLOAT:
        case ImageFormat::FORMAT_RGBA_FLOAT:
            return sizeof(float);
        default:
            throw std::runtime_error("getImageFormatChannelSize: unrecognized image format");
    }
}

uint32_t getImageFormatPixelSize(ImageFormat format)
{
    return getImageFormatChannelCount(format) * getImageFormatChannelSize(format);
}

static bool readImageSTBI(const char *filename, ImageData &image)
{
    int channels, width, height;
    unsigned char* data = nullptr;
    data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data)
        throw std::runtime_error("Failed to read LDR image!");

    switch (channels)
    {
        case 1:
            image.format = FORMAT_R_UINT8;
            break;
        case 2:
            image.format = FORMAT_RG_UINT8;
            break;
        case 3:
            image.format = FORMAT_RGB_UINT8;
            break;
        case 4:
            image.format = FORMAT_RGBA_UINT8;
            break;
        default:
            throw std::runtime_error("Invalid channel count");
    }

    image.width = width;
    image.height = height;

    image.data.assign(data, data + width*height*channels);

    stbi_image_free(data);
    return true;
}

static bool readImageEXR(const char *filename, ImageData &image)
{
    if (IsEXR(filename) != TINYEXR_SUCCESS)
        return false;

    int width, height;
    const char *err = nullptr;
    float* data = nullptr;
    if (LoadEXR(&data, &width, &height, filename, &err) < 0)
        throw std::runtime_error(err);

    // TODO use tinyexr's DeepImage interface for other channel counts!
    image.format = FORMAT_RGBA_FLOAT;
    image.width = width;
    image.height = height;
    image.data.assign(reinterpret_cast<unsigned char*>(data), reinterpret_cast<unsigned char*>(data + 4*width*height));

    free(data);
    return true;
}

bool readImage(const char *filename, ImageData &image)
{
    if (readImageEXR(filename, image))
        return true;
    if (readImageSTBI(filename, image))
        return true;
    return false;
}

void writeImagePNG(const char *filename, const ImageData &image)
{
    uint32_t channelCount = getImageFormatChannelCount(image.format);
    uint32_t channelSize = getImageFormatChannelSize(image.format);
    if (channelSize != 1)
    {
        throw std::runtime_error("Only FORMAT_*_UINT8 can be written to PNG images right now!");
    }

    // stride gives the size of an image row in bytes
    int stride = channelCount * image.width;

    if (stbi_write_png(filename, image.width, image.height, channelCount, image.data.data(), stride) == 0)
    {
        throw std::runtime_error("Failed to write png image!");
    }
}

void writeImageEXR(const char *filename, const ImageData &image, bool halfPrecision)
{
    switch (image.format)
    {
        case FORMAT_RGB_FLOAT:
        case FORMAT_RGBA_FLOAT:
            break;
        default:
            throw std::runtime_error("Only FORMAT_*_FLOAT can be written to EXR images right now!");
    }

    uint32_t channelCount = getImageFormatChannelCount(image.format);

    const char *err = nullptr;
    if (SaveEXR(reinterpret_cast<const float*>(image.data.data()), image.width, image.height,
            channelCount, halfPrecision ? 1 : 0,
            filename, &err) < 0)
    {
        throw std::runtime_error(err);
    }
}

} // end namespace opg
