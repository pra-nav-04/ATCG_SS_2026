add_library( stb STATIC stb_image.cpp stb_image.h stb_image_write.cpp stb_image_write.h )

target_include_directories( stb INTERFACE . )
