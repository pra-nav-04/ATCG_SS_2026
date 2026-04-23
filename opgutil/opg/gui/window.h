#pragma once

#include <cstdint>

#include "opg/opgapi.h"

struct GLFWwindow;
struct ImGuiContext;

namespace opg {

class Window
{
public:
    OPG_API Window(uint32_t width, uint32_t height, const char *window_title);
    OPG_API ~Window();

    OPG_API void beginImGuiFrame();
    OPG_API void endImGuiFrame();

    OPG_API void installImGuiCallbacks();

    inline operator GLFWwindow *() { return window; }
    inline GLFWwindow *getWindowHandle() { return window; }

    inline uint32_t getMainDock() { return main_dock; }

private:
    GLFWwindow *window;
    ImGuiContext *imgui_context;
    uint32_t main_dock;
};

} // end namespace opg
