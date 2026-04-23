#include "Tonemapping.h"

#include <vector>
#include <limits>
#include <iostream>

#include "opg/memory/devicebuffer.h"
#include "opg/exception.h"
#include "opg/imagedata.h"

namespace {

    template <typename T>
    void writeDebugImagePNG(const std::string& path, int width, int height, opg::ImageFormat format, const opg::DeviceBuffer<T>& buffer)
    {
        int channelCount = opg::getImageFormatChannelCount(format);
        std::vector<T> buffer_host(width * height * channelCount);
        buffer.download(buffer_host.data());

        opg::ImageData img_data;
        img_data.data.assign(reinterpret_cast<unsigned char *>(buffer_host.data()), reinterpret_cast<unsigned char *>(buffer_host.data() + buffer_host.size()));
        img_data.format = format;
        img_data.height = height;
        img_data.width = width;

        opg::writeImagePNG(path.c_str(), img_data);
    }

    template <typename T>
    void writeDebugImageEXR(const std::string& path, int width, int height, opg::ImageFormat format, const opg::DeviceBuffer<T>& buffer)
    {
        int channelCount = opg::getImageFormatChannelCount(format);
        std::vector<T> buffer_host(width * height * channelCount);
        buffer.download(buffer_host.data());

        opg::ImageData img_data;
        img_data.data.assign(reinterpret_cast<unsigned char *>(buffer_host.data()), reinterpret_cast<unsigned char *>(buffer_host.data() + buffer_host.size()));
        img_data.format = format;
        img_data.width = width;
        img_data.height = height;

        opg::writeImageEXR(path.c_str(), img_data);
    }
}

opg::ImageData tonemapImage(const opg::ImageData& img, const std::string& mode)
{
    uint32_t inputChannelCount = opg::getImageFormatChannelCount(img.format);

    // allocate device memory for input and output
    opg::DeviceBuffer<float>        hdr_img_buffer(img.width * img.height * inputChannelCount);
    opg::DeviceBuffer<glm::vec3>    rgb_img_buffer(img.width * img.height);
    opg::DeviceBuffer<glm::u8vec3>  ldr_img_buffer(img.width * img.height);

    // upload image to device memory
    hdr_img_buffer.upload(reinterpret_cast<const float*>(img.data.data()));

    // convert to three channels (from arbitrary number (e.g. 4 for RGBA input))
    extractRgbFromMultiChannel(hdr_img_buffer.data(), rgb_img_buffer.data(), inputChannelCount, img.width * img.height);
    CUDA_SYNC_CHECK();

    if (mode == "linear_max")
    {
        // TODO: implement linear tone mapping with maximum color value as scaling factor
//
    }
    else if (mode == "linear_fixed")
    {
        // TODO: implement linear tone mapping with hand picked scaling factor
//
    }
    else if (mode == "gamma_fixed")
    {
        // TODO: implement gamm tone mapping with hand picked gamma
//
    }
    else if (mode == "histogram")
    {
        // allocate necessary device memory
        opg::DeviceBuffer<glm::vec3> hsv_brightness_buffer(img.width * img.height);
        opg::DeviceBuffer<glm::vec3> filtered_rgb_buffer(img.width * img.height);
        opg::DeviceBuffer<glm::vec3> filtered_hsv_brightness_buffer(img.width * img.height);
        opg::DeviceBuffer<float> min_buffer(1);
        opg::DeviceBuffer<float> max_buffer(1);
        uint32_t number_bins = 128;
        opg::DeviceBuffer<uint32_t> bins_buffer(number_bins);
        opg::DeviceBuffer<uint32_t> cum_bins_buffer(number_bins);
        opg::DeviceBuffer<float> accum_probs_buffer(number_bins);

        // upload and initialize data
        float min_init = std::numeric_limits<float>::infinity();
        min_buffer.upload(&min_init);
        float max_init = -std::numeric_limits<float>::infinity();
        max_buffer.upload(&max_init);
        cudaMemset(bins_buffer.data(), 0, sizeof(uint32_t) * number_bins);
        cudaMemset(cum_bins_buffer.data(), 0, sizeof(uint32_t) * number_bins);
        cudaMemset(accum_probs_buffer.data(), 0, sizeof(float) * number_bins);
        CUDA_SYNC_CHECK();

        uint32_t filter_size = 31;
        boxFilterRgb(rgb_img_buffer.data(), filtered_rgb_buffer.data(), filter_size, img.width, img.height);
        CUDA_SYNC_CHECK();

        convertRgbToHsvBrightness(filtered_rgb_buffer.data(), filtered_hsv_brightness_buffer.data(), img.width * img.height);
        CUDA_SYNC_CHECK();

        convertRgbToHsvBrightness(rgb_img_buffer.data(), hsv_brightness_buffer.data(), img.width * img.height);
        CUDA_SYNC_CHECK();

        brightnessMinMax(filtered_hsv_brightness_buffer.data(), min_buffer.data(), max_buffer.data(), img.width * img.height);
        CUDA_SYNC_CHECK();

        // TODO:
        // 1. compute histogram of brightness values
        // 2. compute cumulative sum
        // 3. calculate respective probabilities for brightness values
        // 4. perform histogram mapping on unfiltered image
//

        convertHsvToRgb(hsv_brightness_buffer.data(), rgb_img_buffer.data(), img.width * img.height);
        CUDA_SYNC_CHECK();
    }
    else
    {
        std::cout << "Error: Unknown tonemapping mode '" << mode << "'" << std::endl;
        exit(1);
    }
    
    // convert float to unsigned char
    convertFloatToUint8(reinterpret_cast<float*>(rgb_img_buffer.data()), reinterpret_cast<uint8_t*>(ldr_img_buffer.data()), img.width * img.height * 3);
    CUDA_SYNC_CHECK();

    // assemble results and copy to CPU memory
    std::vector<glm::u8vec3> result_ldr_host(img.width * img.height);
    ldr_img_buffer.download(result_ldr_host.data());

    opg::ImageData result;
    result.data.assign(reinterpret_cast<uint8_t *>(result_ldr_host.data()), reinterpret_cast<uint8_t *>(result_ldr_host.data() + result_ldr_host.size()));
    result.format = opg::ImageFormat::FORMAT_RGB_UINT8;
    result.width = img.width;
    result.height = img.height;

    return result;
}
