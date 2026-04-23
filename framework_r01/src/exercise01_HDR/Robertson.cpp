#include "Robertson.h"

#include <iostream>

#include "opg/glmwrapper.h"
#include "opg/memory/devicebuffer.h"
#include "opg/exception.h"
#include "opg/imagedata.h"

std::pair<opg::ImageData, std::vector<std::vector<float>>> robertson(const std::vector<opg::ImageData> &imgs, const std::vector<float> &exposures, size_t max_iterations)
{
    if (imgs.size() != exposures.size())
    {
        std::cout << "Error: number of images has to match number of exposure times" << std::endl;
        exit(1);
    }

    size_t number_imgs = imgs.size();
    int width = -1;
    int height = -1;
    opg::ImageFormat format = opg::ImageFormat::FORMAT_RGB_UINT8; /// force this image format
    
    for (size_t i = 0; i < imgs.size(); i++)
    {
        if (width == -1)
        {
            width = imgs[i].width;
        }
        else if (width != imgs[i].width)
        {
            std::cout << "Error: input images must have matching dimensions" << std::endl;
            exit(1);
        }

        if (height == -1)
        {
            height = imgs[i].height;
        }
        else if (height != imgs[i].height)
        {
            std::cout << "Error: input images must have matching dimensions" << std::endl;
            exit(1);
        }

        if (format != imgs[i].format)
        {
            std::cout << "Error: input images must have matching image formats" << std::endl;
            exit(1);
        }
    }

    std::cout << "upload and prepare data" << std::endl;

    // upload data to device
    opg::DeviceBuffer<glm::u8vec3> imgs_buffer(number_imgs * width * height);

    for (size_t i = 0; i < imgs.size(); i++)
    {
        imgs_buffer.uploadSub(reinterpret_cast<const glm::u8vec3*>(imgs[i].data.data()), width * height, i * width * height);
    }

    // split image into different channels for optimized memory access
    std::vector<opg::DeviceBuffer<uint8_t>> channel_buffers;
    for (size_t i = 0; i < 3; i++)
    {
        channel_buffers.emplace_back(number_imgs * width * height);
    }

    // work on channels separately. spliting them here makes memory access more efficient.
    splitChannels(imgs_buffer.data(), channel_buffers[0].data(), channel_buffers[1].data(), channel_buffers[2].data(), number_imgs * width * height);
    CUDA_SYNC_CHECK();

    opg::DeviceBuffer<float> exposures_buffer(number_imgs);
    exposures_buffer.upload(exposures.data());

    // allocate auxilliary buffers for algorithm
    opg::DeviceBuffer<uint32_t> counters_buffer(256);
    opg::DeviceBuffer<float> weights_buffer(number_imgs * width * height);
    opg::DeviceBuffer<bool> underexposed_buffer(width * height);
    opg::DeviceBuffer<float> x_nom_buffer(width * height);
    opg::DeviceBuffer<float> x_denom_buffer(width * height);
    opg::DeviceBuffer<float> x_buffer(width * height);
    opg::DeviceBuffer<float> I_unnorm_buffer(256);
    opg::DeviceBuffer<float> I_buffer(256);

    std::cout << "run precalculations" << std::endl;

    // allocate memory to store results in
    std::vector<opg::DeviceBuffer<float>> result_x_channel_buffers;
    for (size_t i = 0; i < 3; i++)
    {
        result_x_channel_buffers.emplace_back(width * height);
    }

    std::vector<opg::DeviceBuffer<float>> result_I_channel_buffers;
    for (size_t i = 0; i < 3; i++)
    {
        result_I_channel_buffers.emplace_back(256);
    }

    // run algorithm separately for channels
    for (size_t i = 0; i < channel_buffers.size(); i++)
    {
        std::cout << "process channel " << i << std::endl;

        // reset buffers where necessary
        cudaMemset(counters_buffer.data(), 0, 256 * sizeof(uint32_t));
        cudaMemset(underexposed_buffer.data(), 0, width * height * sizeof(bool));
        CUDA_SYNC_CHECK();

        // mask out pixels which are in general underexposed
        calcMask(channel_buffers[i].data(), underexposed_buffer.data(), number_imgs * width * height, width * height);
        CUDA_SYNC_CHECK();

        // count occurences of different numbers
        countValues(channel_buffers[i].data(), underexposed_buffer.data(), counters_buffer.data(), number_imgs * width * height, width * height);
        CUDA_SYNC_CHECK();

        // calculate weights (low for extreme values)
        calcWeights(channel_buffers[i].data(), weights_buffer.data(), number_imgs * width * height);
        CUDA_SYNC_CHECK();

        // init inverse CRF linearly
        initInvCrf(I_buffer.data(), 256);
        CUDA_SYNC_CHECK();

        for (size_t iteration = 0; iteration < max_iterations; iteration++)
        {
            // TODO:
            // 1. reset/initialize your buffers for this iteration
            // 2. calculate the irradiance estimate
            // 3. calculate the new inverse CRF estimate
            // 4. normalize the inverse CRF
//
        }

        // copy results for this channel to not overwrite them with next channel results
        cudaMemcpy(result_I_channel_buffers[i].data(), I_buffer.data(), sizeof(float) * 256, cudaMemcpyDeviceToDevice);
        cudaMemcpy(result_x_channel_buffers[i].data(), x_buffer.data(), sizeof(float) * width * height, cudaMemcpyDeviceToDevice);
    }

    std::cout << "download and assemble results" << std::endl;

    // assemble results and copy to CPU memory
    opg::DeviceBuffer<glm::f32vec3> result_hdr_buffer(width * height);
    mergeChannels(result_hdr_buffer.data(), result_x_channel_buffers[0].data(), result_x_channel_buffers[1].data(), result_x_channel_buffers[2].data(), width * height);
    CUDA_SYNC_CHECK();
    std::vector<glm::f32vec3> result_hdr_host(width * height);
    result_hdr_buffer.download(result_hdr_host.data());

    opg::ImageData result_hdr;
    result_hdr.data.assign(reinterpret_cast<unsigned char *>(result_hdr_host.data()), reinterpret_cast<unsigned char *>(result_hdr_host.data() + result_hdr_host.size()));
    result_hdr.format = opg::ImageFormat::FORMAT_RGB_FLOAT;
    result_hdr.width = width;
    result_hdr.height = height;

    std::vector<std::vector<float>> crfs;
    for (size_t i = 0; i < 3; i++)
    {
        std::vector<float> crf(256);
        result_I_channel_buffers[i].download(crf.data());
        crfs.push_back(std::move(crf));
    }

    return {result_hdr, crfs};
}