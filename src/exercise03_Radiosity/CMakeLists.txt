# Collect all source files in this directory
file(GLOB_RECURSE SOURCE_FILES "*.h" "*.cuh" "*.cu" "*.cpp")
# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add target executable
OPTIX_add_executable(${TARGET_NAME} target_name ${SOURCE_FILES})
