#pragma once

#include <glad/glad.h>

#include <cstdint>
#include <string>

#include "opg/imagedata.h"

#include "opg/opgapi.h"
#include "opg/glmwrapper.h"

namespace opg {

class GLDisplay
{
public:
    OPG_API GLDisplay(ImageFormat format = ImageFormat::FORMAT_RGBA_UINT8);
    OPG_API ~GLDisplay();

    GLDisplay(const GLDisplay &) = delete;
    OPG_API GLDisplay(GLDisplay &&other);

    GLDisplay &operator = (const GLDisplay &) = delete;
    OPG_API GLDisplay &operator = (GLDisplay &&other);

    OPG_API void update(uint32_t image_width, uint32_t image_height, GLuint pbo);

    OPG_API void renderFramebuffer();
    OPG_API void renderGui();

    inline void setGuiWindowName(std::string name) { m_gui_window_name = name; }
    inline void setGuiCanClose(bool gui_can_close) { m_gui_can_close = gui_can_close; }
    inline bool isGuiWindowVisible() const { return m_gui_visible; }
    inline void setGuiWindowVisible(bool gui_visible) { m_gui_visible = gui_visible; }
    inline bool isGuiWindowInFocus() const { return m_gui_visible && m_gui_focus; }

    inline void setGuiWindowOverrideSize(bool override) { m_gui_override_size = override; }
    inline void setGuiWindowSize(const glm::uvec2 &size) { m_gui_width = size.x; m_gui_height = size.y; }
    inline glm::uvec2 getGuiWindowSize() const { return glm::uvec2(m_gui_width, m_gui_height); }

private:
    GLuint   m_render_tex = 0u;
    GLuint   m_program = 0u;
    GLint    m_render_tex_uniform_loc = -1;
    GLuint   m_vertex_array = 0u;
    GLuint   m_quad_vertex_buffer = 0u;

    GLint    m_internal_format;
    GLenum   m_input_format;
    GLenum   m_input_type;

    ImageFormat m_image_format;

    // ImGui "window" properties
    bool        m_gui_visible;
    bool        m_gui_focus;
    bool        m_gui_can_close;
    bool        m_gui_override_size;
    uint32_t    m_gui_width;
    uint32_t    m_gui_height;
    std::string m_gui_window_name;
};

} // end namespace opg
