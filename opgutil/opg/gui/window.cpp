#include "opg/gui/window.h"
#include "opg/exception.h"

#include <iostream>

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

static void errorCallback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}


static void initGL()
{
    OPG_CHECK_MSG( gladLoadGL(), "Failed to initialze GL" );

    GL_CHECK( glClearColor(0.212f, 0.271f, 0.31f, 1.0f) );
    GL_CHECK( glClear(GL_COLOR_BUFFER_BIT) );
}

class GLFWInstanceGuard
{
public:
    GLFWInstanceGuard() = default;

    GLFWInstanceGuard(const GLFWInstanceGuard &) = delete;
    GLFWInstanceGuard(GLFWInstanceGuard &&) = delete;

    GLFWInstanceGuard &operator = (const GLFWInstanceGuard &) = delete;
    GLFWInstanceGuard &operator = (GLFWInstanceGuard &&) = delete;

    ~GLFWInstanceGuard()
    {
        if (initialized)
            glfwTerminate();
    }

    void init()
    {
        if (!initialized)
        {
            glfwSetErrorCallback(errorCallback);
            OPG_CHECK_MSG( glfwInit(), "Failed to initialize GLFW" );
            initialized = true;
        }
    }

private:
    bool initialized = false;
};

static GLFWInstanceGuard glfwInstance;


static GLFWwindow* initGLFW(int width, int height, const char *window_title)
{
    glfwInstance.init();

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );

    GLFWwindow* window = glfwCreateWindow(width, height, window_title, nullptr, nullptr);
    OPG_ASSERT_MSG( window, "Failed to create GLFW window" );

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // No vsync

    return window;
}

static ImGuiContext *initImGui(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGuiContext *context = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui_ImplGlfw_InitForOpenGL( window, false );
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
    io.Fonts->AddFontDefault();

    ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 0.8f;

    return context;
}

namespace opg {

Window::Window(uint32_t width, uint32_t height, const char* window_title)
{
    window = initGLFW(width, height, window_title);
    initGL();
    imgui_context = initImGui(window);

    glfwShowWindow(window);
}

Window::~Window()
{
    ImGui::SetCurrentContext(imgui_context);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
}


void Window::beginImGuiFrame()
{
    ImGui::SetCurrentContext(imgui_context);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable)
        main_dock = ImGui::DockSpaceOverViewport(nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
}

void Window::endImGuiFrame()
{
    ImGui::EndFrame();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

    // Update and Render additional Platform Windows
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }

    // OpenGL context might have switched to other window due to ImGUI viewports
    glfwMakeContextCurrent(window);
}

void Window::installImGuiCallbacks()
{
    ImGui::SetCurrentContext(imgui_context);
    ImGui_ImplGlfw_InstallCallbacks(window);
}

} // end namespace opg
