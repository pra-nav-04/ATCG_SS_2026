find_package( OpenGL REQUIRED )

add_library( glad SHARED
  KHR/khrplatform.h
  glad/glad.c
  glad/glad.h
)
# Add definitions that allow the symbols to be resolved across shared library boundaries.
# (These definitions are not required if we link statically)
target_compile_definitions( glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC  GLAD_GLAPI_EXPORT )
target_include_directories( glad PUBLIC . )
target_link_libraries( glad PUBLIC ${OPENGL_LIBRARIES} )
