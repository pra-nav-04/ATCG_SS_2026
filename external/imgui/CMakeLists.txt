
file( GLOB IMGUI_SOURCES *.h *.cpp )

add_library( imgui STATIC ${IMGUI_SOURCES} )

target_include_directories( imgui INTERFACE .. )
target_compile_definitions( imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD )
target_link_libraries( imgui glfw glad ${OPENGL_LIBRARIES} )
