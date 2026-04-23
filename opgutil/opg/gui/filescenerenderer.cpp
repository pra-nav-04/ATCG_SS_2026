#include "opg/gui/filescenerenderer.h"

#include "opg/scene/scene.h"
#include "opg/kernels/kernels.h"
#include "opg/imagedata.h"

#include <GLFW/glfw3.h>
#include <fmt/core.h>

namespace opg {

FileSceneRenderer::FileSceneRenderer(Scene *scene, uint32_t width, uint32_t height) :
    m_scene { scene },
    m_width { width },
    m_height { height }
{
}

FileSceneRenderer::~FileSceneRenderer()
{
}

void FileSceneRenderer::run(const std::string &output_filename, uint32_t sample_count)
{
    DeviceBuffer<glm::vec3> hdr_buffer(m_width * m_height);
    for (uint32_t i = 0; i < sample_count; ++i)
    {
        fmt::print("\rRendering... [ {}/{} ]", i, sample_count);
        std::cout << std::flush;
        m_scene->traceRays(make_tensor_view(hdr_buffer.view(), m_height, m_width));
    }
    /*
    auto progress_callback = [](float progress)
    {
        fmt::print("\rRendering... [ {:5.02f}% ]", progress*100.0f);
        std::cout << std::flush;
    };
    m_scene->traceRays(make_tensor_view(hdr_buffer.view(), m_height, m_width), sample_count, progress_callback);
    */
    std::cout << std::endl;

    // Apply tonemapping to PNG output files
    if (output_filename.size() >= 4 && output_filename.substr(output_filename.size() - 4) == ".png")
    {
        auto ldr_buffer = opg::DeviceBuffer<glm::u8vec3>(m_width * m_height);

        // Apply tonemapping to produce final LDR image
        opg::BufferView<glm::vec3> hdr_buffer_view = hdr_buffer.view();
        opg::BufferView<glm::u8vec3> ldr_buffer_view = ldr_buffer.view();
        opg::tonemap_srgb(hdr_buffer_view, ldr_buffer_view);
        CUDA_SYNC_CHECK();

        std::vector<glm::u8vec3> ldr_data(m_width * m_height);
        ldr_buffer.download(ldr_data.data());

        opg::ImageData image;
        image.data.assign(reinterpret_cast<unsigned char*>(ldr_data.data()), reinterpret_cast<unsigned char*>(ldr_data.data() + m_width*m_height));
        image.format = opg::ImageFormat::FORMAT_RGB_UINT8;
        image.width = m_width;
        image.height = m_height;

        opg::writeImagePNG(output_filename.c_str(), image);
        std::cout << "Wrote PNG image to " << output_filename << std::endl;
    }
    else // if (output_filename.ends_with(".exr"))
    {
        std::vector<glm::vec3> hdr_data(m_width * m_height);
        hdr_buffer.download(hdr_data.data());

        opg::ImageData image;
        image.data.assign(reinterpret_cast<unsigned char*>(hdr_data.data()), reinterpret_cast<unsigned char*>(hdr_data.data() + m_width*m_height));
        image.format = opg::ImageFormat::FORMAT_RGB_FLOAT;
        image.width = m_width;
        image.height = m_height;

        opg::writeImageEXR(output_filename.c_str(), image);
        std::cout << "Wrote EXR image to " << output_filename << std::endl;
    }
}

} // end namespace opg
