#include "opg/exception.h"
#include "opg/gui/gldisplay.h"

#include <imgui/imgui.h>

namespace opg {

//-----------------------------------------------------------------------------
//
// Helper functions
//
//-----------------------------------------------------------------------------

static GLuint createGLShader(const char *source, GLuint shader_type)
{
    GLuint shader = glCreateShader(shader_type);
    {
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        GLint is_compiled = 0;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &is_compiled );
        if( is_compiled == GL_FALSE )
        {
            GLint max_length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

            std::string info_log( max_length, '\0' );
            glGetShaderInfoLog(shader, max_length, nullptr, info_log.data());

            glDeleteShader(shader);

            std::ostringstream ss;
            ss << "Compilation of shader failed: " << info_log << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

    GL_CHECK_ERRORS();

    return shader;
}


static GLuint createGLProgram(const char *vert_source, const char *frag_source)
{
    GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
    GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE)
    {
        GLint max_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

        std::string info_log(max_length, '\0');
        glGetProgramInfoLog(program, max_length, nullptr, info_log.data());

        glDeleteProgram(program);
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        std::ostringstream ss;
        ss << "Linking of program failed: " << info_log << std::endl;
        throw std::runtime_error(ss.str());
    }

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    GL_CHECK_ERRORS();

    return program;
}


static GLint getGLUniformLocation(GLuint program, const std::string& name)
{
    GLint loc = glGetUniformLocation( program, name.c_str() );
    OPG_ASSERT_MSG( loc != -1, "Failed to get uniform loc for '" + name + "'" );
    return loc;
}


static void getFormatParameters(ImageFormat image_format, GLint &internal_format, GLenum &input_format, GLenum &input_type)
{
    switch (image_format)
    {
        case ImageFormat::FORMAT_R_UINT8:
        {
            internal_format = GL_R8;
            input_format = GL_RED;
            input_type = GL_UNSIGNED_BYTE;
        } break;
        case ImageFormat::FORMAT_RG_UINT8:
        {
            internal_format = GL_RG8;
            input_format = GL_RG;
            input_type = GL_UNSIGNED_BYTE;
        } break;
        case ImageFormat::FORMAT_RGB_UINT8:
        {
            internal_format = GL_RGB8;
            input_format = GL_RGB;
            input_type = GL_UNSIGNED_BYTE;
        } break;
        case ImageFormat::FORMAT_RGBA_UINT8:
        {
            internal_format = GL_RGBA8;
            input_format = GL_RGBA;
            input_type = GL_UNSIGNED_BYTE;
        } break;
        case ImageFormat::FORMAT_RGB_FLOAT:
        {
            internal_format = GL_RGB32F;
            input_format = GL_RGB;
            input_type = GL_FLOAT;
        } break;
        case ImageFormat::FORMAT_RGBA_FLOAT:
        {
            internal_format = GL_RGBA32F;
            input_format = GL_RGBA;
            input_type = GL_FLOAT;
        } break;
        default:
            throw std::runtime_error("Unknown buffer format");
    }
}

//-----------------------------------------------------------------------------
//
// GLDisplay implementation
//
//-----------------------------------------------------------------------------

static const char s_vert_source[] = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, -vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

static const char s_frag_source[] = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";



GLDisplay::GLDisplay(ImageFormat image_format) :
    m_image_format { image_format },
    m_gui_visible { false },
    m_gui_focus { true },
    m_gui_can_close { false },
    m_gui_override_size { false },
    m_gui_width { 0 },
    m_gui_height { 0 },
    m_gui_window_name { "Viewport" }
{
    getFormatParameters(m_image_format, m_internal_format, m_input_format, m_input_type);

    // Prepare shader program

    m_program = createGLProgram( s_vert_source, s_frag_source );
    m_render_tex_uniform_loc = 0;//getGLUniformLocation( m_program, "render_tex");

    // Prepare texture

    GL_CHECK( glGenTextures( 1, &m_render_tex ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );

    GL_CHECK( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) );
    GL_CHECK( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) );
    GL_CHECK( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
    GL_CHECK( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );

    GL_CHECK( glBindTexture( GL_TEXTURE_2D, 0 ) );

    // Prepare vertex data

    GL_CHECK( glGenVertexArrays(1, &m_vertex_array ) );
    GL_CHECK( glBindVertexArray( m_vertex_array ) );

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    GL_CHECK( glGenBuffers(1, &m_quad_vertex_buffer) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer) );
    GL_CHECK( glBufferData(GL_ARRAY_BUFFER,
        sizeof(g_quad_vertex_buffer_data),
        g_quad_vertex_buffer_data,
        GL_STATIC_DRAW
    ) );

    // 1st attribute buffer : vertices
    GL_CHECK( glEnableVertexAttribArray(0) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer) );
    GL_CHECK( glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    ) );

    GL_CHECK( glBindVertexArray( 0 ) );

    GL_CHECK_ERRORS();
}

GLDisplay::~GLDisplay()
{
    glDeleteProgram(m_program);
    glDeleteTextures(1, &m_render_tex);
    glDeleteVertexArrays(1, &m_vertex_array);
    glDeleteBuffers(1, &m_quad_vertex_buffer);
}

GLDisplay::GLDisplay(GLDisplay &&other)
{
    m_render_tex                = other.m_render_tex;
    m_program                   = other.m_program;
    m_render_tex_uniform_loc    = other.m_render_tex_uniform_loc;
    m_vertex_array              = other.m_vertex_array;
    m_quad_vertex_buffer        = other.m_quad_vertex_buffer;

    other.m_render_tex          = 0u;
    other.m_program             = 0u;
    other.m_vertex_array        = 0u;
    other.m_quad_vertex_buffer  = 0u;


    m_internal_format   = other.m_internal_format;
    m_input_format      = other.m_input_format;
    m_input_type        = other.m_input_type;
    m_image_format      = other.m_image_format;

    m_gui_visible       = other.m_gui_visible;
    m_gui_focus         = other.m_gui_focus;
    m_gui_can_close     = other.m_gui_can_close;
    m_gui_override_size = other.m_gui_override_size;
    m_gui_width         = other.m_gui_width;
    m_gui_height        = other.m_gui_height;
    m_gui_window_name   = std::move(other.m_gui_window_name);
}

GLDisplay &GLDisplay::operator = (GLDisplay &&other)
{
    m_render_tex                = other.m_render_tex;
    m_program                   = other.m_program;
    m_render_tex_uniform_loc    = other.m_render_tex_uniform_loc;
    m_vertex_array              = other.m_vertex_array;
    m_quad_vertex_buffer        = other.m_quad_vertex_buffer;

    other.m_render_tex          = 0u;
    other.m_program             = 0u;
    other.m_vertex_array        = 0u;
    other.m_quad_vertex_buffer  = 0u;


    m_internal_format   = other.m_internal_format;
    m_input_format      = other.m_input_format;
    m_input_type        = other.m_input_type;
    m_image_format      = other.m_image_format;

    m_gui_visible       = other.m_gui_visible;
    m_gui_focus         = other.m_gui_focus;
    m_gui_can_close     = other.m_gui_can_close;
    m_gui_override_size = other.m_gui_override_size;
    m_gui_width         = other.m_gui_width;
    m_gui_height        = other.m_gui_height;
    m_gui_window_name   = std::move(other.m_gui_window_name);

    return *this;
}

void GLDisplay::update(uint32_t image_width, uint32_t image_height, GLuint pbo)
{
    GL_CHECK( glBindTexture(GL_TEXTURE_2D, m_render_tex) );
    GL_CHECK( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo) );

    size_t elmt_size = getImageFormatPixelSize(m_image_format);
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, m_internal_format, image_width, image_height, 0, m_input_format, m_input_type, nullptr);

    GL_CHECK( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) );
    GL_CHECK( glBindTexture(GL_TEXTURE_2D, 0) );
}

void GLDisplay::renderFramebuffer()
{
    GL_CHECK( glUseProgram(m_program) );

    // Bind our texture in Texture Unit 0
    GL_CHECK( glActiveTexture(GL_TEXTURE0) );
    GL_CHECK( glBindTexture(GL_TEXTURE_2D, m_render_tex) );
    GL_CHECK( glUniform1i(m_render_tex_uniform_loc, 0) );

    // Bind vertex array
    GL_CHECK( glBindVertexArray( m_vertex_array ) );

    // Draw the triangles!
    GL_CHECK( glDrawArrays(GL_TRIANGLES, 0, 6) ); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK( glBindVertexArray(0) );
    GL_CHECK( glBindTexture(GL_TEXTURE_2D, 0) );

    GL_CHECK_ERRORS();
}

void GLDisplay::renderGui()
{
    if (m_gui_visible)
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar;
        if (m_gui_override_size)
            window_flags |= ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize;
        if (ImGui::Begin(m_gui_window_name.c_str(), m_gui_can_close ? &m_gui_visible : nullptr, window_flags))
        {
            ImVec2 view_size = ImGui::GetContentRegionAvail();
            if (m_gui_override_size)
                view_size = ImVec2(m_gui_width, m_gui_height);

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
            ImGui::ImageButton("viewport_img", reinterpret_cast<ImTextureID>(static_cast<size_t>(m_render_tex)), view_size);
            ImGui::PopStyleVar();
            ImGuiID viewportID = ImGui::GetItemID();
            ImGui::SetItemDefaultFocus();

            m_gui_focus = ImGui::IsItemFocused();

            // Remember size of the view window
            m_gui_width = static_cast<uint32_t>(view_size.x);
            m_gui_height = static_cast<uint32_t>(view_size.y);

            ImGui::End();
        }
        ImGui::PopStyleVar();
    }
    else
    {
        m_gui_focus = false;
    }
}

} // namespace opg
