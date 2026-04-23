add_library( tinyobj STATIC tiny_obj_loader.cpp tiny_obj_loader.h )

target_include_directories( tinyobj INTERFACE . )

