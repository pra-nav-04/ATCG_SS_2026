#pragma once

#include <memory>

#include "opg/memory/outputbuffer.h"
#include "opg/memory/devicebuffer.h"
#include "opg/glmwrapper.h"
#include "opg/gui/window.h"
#include "opg/gui/gldisplay.h"
#include "opg/gui/statsdisplay.h"
#include "opg/gui/cameracontroller.h"
#include "opg/opgapi.h"

namespace opg {

class Scene;

class InteractiveSceneRenderer
{
public:
    OPG_API InteractiveSceneRenderer(Scene *scene, uint32_t width, uint32_t height, const char *window_title);
    OPG_API ~InteractiveSceneRenderer();

    OPG_API void run();

private:
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y);
    static void keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods);
    static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll);


    void updateState();
    void launchFrame();
    void render();
    void renderGui();

    void saveScreenshot(const std::string &output_filename);

    // Get the size of the framebuffer for the raytracing image.
    // The chosen size depends on whether the image is shown in the host window or in an ImGui viewport.
    glm::uvec2 getDesiredImageSize();

private:
    Window            m_window;
    GLDisplay         m_image_display;
    StatsDisplay      m_stats_display;
    CameraController  m_camera_controller;

    OutputBuffer<glm::u8vec3> m_output_buffer;
    DeviceBuffer<glm::vec3>   m_hdr_buffer;

    Scene*            m_scene;


    // Window state
    bool              m_resize_dirty  = false;
    uint32_t          m_width;
    uint32_t          m_height;


    // Mouse state
    int32_t           m_mouse_button = -1;

    // Render into an ImGui window, or into the os window.
    bool              m_use_gui_window = false;
    bool              m_override_image_size = false;
    uint32_t          m_override_image_width = 512;
    uint32_t          m_override_image_height = 512;
};

} // end namespace opg
