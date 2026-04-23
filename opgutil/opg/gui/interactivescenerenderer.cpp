#include "opg/gui/interactivescenerenderer.h"

#include "opg/scene/scene.h"
#include "opg/kernels/kernels.h"

#include <filesystem>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <nfd.hpp>

namespace opg {

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

void InteractiveSceneRenderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    InteractiveSceneRenderer* context = reinterpret_cast<InteractiveSceneRenderer*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !context->m_image_display.isGuiWindowInFocus())
    {
        context->m_mouse_button = -1;
        return;
    }

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        context->m_mouse_button = button;

        if (button == GLFW_MOUSE_BUTTON_LEFT)
            context->m_camera_controller.setViewMode(CameraController::LookAtFixed);
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            context->m_camera_controller.setViewMode(CameraController::EyeFixed);

        context->m_camera_controller.startTracking(static_cast<float>(xpos), static_cast<float>(ypos));
    }
    else
    {
        context->m_mouse_button = -1;
    }
}


void InteractiveSceneRenderer::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    InteractiveSceneRenderer* context = reinterpret_cast<InteractiveSceneRenderer*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !context->m_image_display.isGuiWindowInFocus())
    {
        return;
    }

    if (context->m_mouse_button != -1)
    {
        context->m_camera_controller.updateTracking(static_cast<float>(xpos), static_cast<float>(ypos));
    }
}


void InteractiveSceneRenderer::windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    InteractiveSceneRenderer* context = reinterpret_cast<InteractiveSceneRenderer*>(glfwGetWindowUserPointer(window));
    context->m_width  = res_x;
    context->m_height = res_y;

    // Only if image is shown in native window!
    //context->m_camera_controller.resizeViewport(res_x, res_y);
}


void InteractiveSceneRenderer::keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t mods)
{
    InteractiveSceneRenderer* context = reinterpret_cast<InteractiveSceneRenderer*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureKeyboard && !context->m_image_display.isGuiWindowInFocus())
    {
        return;
    }

    if (key == GLFW_KEY_ESCAPE)
    {
        if (action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        if (action == GLFW_PRESS)
        {
            // toggle UI draw
            context->m_stats_display.toggleVisibility();
        }
    }
    else
    {
        // Forward to camera controller
        context->m_camera_controller.keyEvent(key, action, mods);
    }
}


void InteractiveSceneRenderer::scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    InteractiveSceneRenderer* context = reinterpret_cast<InteractiveSceneRenderer*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !context->m_image_display.isGuiWindowInFocus())
    {
        return;
    }

    context->m_camera_controller.wheelEvent(static_cast<float>(yscroll));
}


//
//
//


InteractiveSceneRenderer::InteractiveSceneRenderer(Scene *scene, uint32_t width, uint32_t height, const char *window_title) :
    m_window { width, height, window_title },
    m_image_display { ImageFormat::FORMAT_RGB_UINT8 },
    m_stats_display { &m_window },
    m_scene { scene },
    m_output_buffer { OutputBufferType::GL_INTEROP, width, height },
    m_hdr_buffer { width * height },
    m_width { width },
    m_height { height }
{
    Camera *camera_component = m_scene->getRayGenerator()->getCamera();
    m_camera_controller.setCamera(camera_component);

    glfwSetMouseButtonCallback( m_window, &InteractiveSceneRenderer::mouseButtonCallback );
    glfwSetCursorPosCallback  ( m_window, &InteractiveSceneRenderer::cursorPosCallback   );
    glfwSetWindowSizeCallback ( m_window, &InteractiveSceneRenderer::windowSizeCallback  );
    glfwSetKeyCallback        ( m_window, &InteractiveSceneRenderer::keyCallback         );
    glfwSetScrollCallback     ( m_window, &InteractiveSceneRenderer::scrollCallback      );
    glfwSetWindowUserPointer  ( m_window, this                                           );

    // After user callbacks have been registered...
    m_window.installImGuiCallbacks();

    NFD::Init();
}

InteractiveSceneRenderer::~InteractiveSceneRenderer()
{
    NFD::Quit();
}

void InteractiveSceneRenderer::updateState()
{
    // TODO notify scene about camera

    // TODO notify scene about output image size?

    // Make sure that output buffer has correct size
    glm::uvec2 image_size = getDesiredImageSize();
    if (image_size.x != m_output_buffer.width() || image_size.y != m_output_buffer.height())
    {
        m_output_buffer.resize(image_size.x, image_size.y);
        m_hdr_buffer.alloc(image_size.x * image_size.y);

        m_camera_controller.resizeViewport(image_size.x, image_size.y);
    }
}

void InteractiveSceneRenderer::launchFrame()
{
    glm::u8vec3* result_buffer_data = m_output_buffer.mapCUDA();

    // TODO
    m_scene->traceRays(make_tensor_view(m_hdr_buffer.view(), m_output_buffer.height(), m_output_buffer.width()));

    // Apply tonemapping to produce final LDR image
    BufferView<glm::vec3> hdr_buffer_view = m_hdr_buffer.view();
    BufferView<glm::u8vec3> output_buffer_view;
    output_buffer_view.data = reinterpret_cast<std::byte*>(result_buffer_data);
    output_buffer_view.count = m_output_buffer.width() * m_output_buffer.height();
    tonemap_srgb(hdr_buffer_view, output_buffer_view);
    CUDA_SYNC_CHECK();

    m_output_buffer.unmapCUDA();
    CUDA_SYNC_CHECK();
}

void InteractiveSceneRenderer::run()
{
    //
    // Render loop
    //
    auto lastTick = std::chrono::steady_clock::now();
    do
    {
        auto t0 = StatsDisplay::clock::now();

        glfwPollEvents();
        updateState();

        auto currentTick = std::chrono::steady_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(currentTick-lastTick).count();
        lastTick = currentTick;
        m_camera_controller.tick(dt);

        auto t1 = StatsDisplay::clock::now();
        StatsDisplay::duration state_update_time = t1 - t0;
        t0 = t1;

        launchFrame();

        t1 = StatsDisplay::clock::now();
        StatsDisplay::duration render_time = t1 - t0;
        t0 = t1;

        render();
        renderGui();

        t1 = StatsDisplay::clock::now();
        StatsDisplay::duration display_time = t1 - t0;

        m_stats_display.updateStats(state_update_time, render_time, display_time);

        glfwSwapBuffers(m_window);
    }
    while( !glfwWindowShouldClose(m_window) );
}

void InteractiveSceneRenderer::render()
{
    m_image_display.update(m_output_buffer.width(), m_output_buffer.height(), m_output_buffer.getPBO());

    // Render to window framebuffer
    GL_CHECK( glBindFramebuffer(GL_FRAMEBUFFER, 0) );
    GL_CHECK( glViewport(0, 0, m_width, m_height) );
    // Always clear...
    GL_CHECK( glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) );

    if (!m_use_gui_window)
    {
        // If we don't use a GUI window to display the output, render to framebuffer now.
        m_image_display.renderFramebuffer();
    }
}

void InteractiveSceneRenderer::renderGui()
{
    m_window.beginImGuiFrame();

    m_image_display.setGuiWindowVisible(m_use_gui_window);
    m_image_display.renderGui();
    m_stats_display.renderGui();

    ImGui::Begin("Config", nullptr, ImGuiWindowFlags_MenuBar);

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Take Screenshot"))
            {
                std::vector<nfdu8filteritem_t> filters {
                    {
                        "OpenEXR",
                        "*.exr"
                    },
                    {
                        "PNG",
                        "*.png"
                    }
                };
                NFD::UniquePathU8 output_filename;
                nfdresult_t result = NFD::SaveDialog(output_filename,
                                            /*const nfdnfilteritem_t* filterList =*/ filters.data(),
                                            /*nfdfiltersize_t filterCount =*/ filters.size(),
                                            /*const nfdnchar_t* defaultPath =*/ std::filesystem::current_path().string().c_str(),
                                            /*const nfdnchar_t* defaultName =*/ "screenshot.exr");
                if (result == NFD_ERROR)
                {
                    throw std::runtime_error(NFD::GetError());
                }
                else if (result == NFD_OKAY)
                {
                    saveScreenshot(output_filename.get());
                }
                else // if (result == NFD_CANCEL)
                {
                    // NOP
                }
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    if (ImGui::CollapsingHeader("Display"))
    {
        ImGui::Checkbox("Use GUI Window", &m_use_gui_window);
        ImGui::Checkbox("Override Image Size", &m_override_image_size);
        ImGui::InputInt2("Override Image Size", reinterpret_cast<int32_t*>(&m_override_image_width));
    }

    bool scene_changed = false;
    auto show_component_gui = [&](auto *comp)
    {
        if (comp->getName().empty() || !comp->hasGui())
            return;
        if (ImGui::TreeNode(comp, "%s", comp->getName().c_str()))
        {
            scene_changed |= comp->renderGui();
            ImGui::TreePop();
        }
    };

    if (ImGui::CollapsingHeader("Integrator"))
    {
        scene_changed |= m_scene->getRayGenerator()->renderGui();
    }

    if (ImGui::CollapsingHeader("BSDFs"))
    {
        m_scene->traverseSceneComponents<BSDF>(show_component_gui);
    }

    if (ImGui::CollapsingHeader("Emitters"))
    {
        m_scene->traverseSceneComponents<Emitter>(show_component_gui);
    }

    // Add gui for all the scene objects!
    if (ImGui::CollapsingHeader("Scene"))
    {
        m_scene->traverseSceneComponents<SceneComponent>(show_component_gui);
    }

    ImGui::End();

    m_window.endImGuiFrame();

    if (scene_changed)
    {
        m_scene->getRayGenerator()->onSceneChanged();
    }
}

glm::uvec2 InteractiveSceneRenderer::getDesiredImageSize()
{
    if (m_override_image_size)
    {
        return glm::uvec2(m_override_image_width, m_override_image_height);
    }

    if (m_use_gui_window)
    {
        glm::uvec2 gui_window_size = m_image_display.getGuiWindowSize();
        if (gui_window_size.x > 0 && gui_window_size.y > 0)
            return gui_window_size;
    }

    // Default to native framebuffer size
    return glm::uvec2(m_width, m_height);
}


void InteractiveSceneRenderer::saveScreenshot(const std::string &output_filename)
{
    uint32_t output_width = m_output_buffer.width();
    uint32_t output_height = m_output_buffer.height();
    if (output_filename.size() >= 4 && output_filename.substr(output_filename.size() - 4) == ".png")
    {
        glm::u8vec3* ldr_data = m_output_buffer.getHostPointer();

        opg::ImageData image;
        image.data.assign(reinterpret_cast<unsigned char*>(ldr_data), reinterpret_cast<unsigned char*>(ldr_data + output_width*output_height));
        image.format = opg::ImageFormat::FORMAT_RGB_UINT8;
        image.width = output_width;
        image.height = output_height;

        opg::writeImagePNG(output_filename.c_str(), image);
        std::cout << "Wrote PNG image to " << output_filename << std::endl;
    }
    else // if (output_filename.ends_with(".exr"))
    {
        std::vector<glm::vec3> hdr_data(output_width * output_height);
        m_hdr_buffer.download(hdr_data.data());

        opg::ImageData image;
        image.data.assign(reinterpret_cast<unsigned char*>(hdr_data.data()), reinterpret_cast<unsigned char*>(hdr_data.data() + output_width*output_height));
        image.format = opg::ImageFormat::FORMAT_RGB_FLOAT;
        image.width = output_width;
        image.height = output_height;

        opg::writeImageEXR(output_filename.c_str(), image);
        std::cout << "Wrote EXR image to " << output_filename << std::endl;
    }
}

} // end namespace opg
