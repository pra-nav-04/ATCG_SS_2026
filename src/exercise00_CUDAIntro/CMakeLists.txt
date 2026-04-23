
# By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
# into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
# linked against the application such that we can simply call the functions defined in the kernels.cu file.
# The following command would flag the given source files to be compiled into "normal" .obj files, but the same is
# achieved using a corresponding #pragma definition in the respective source file.
#set_source_files_properties(kernels.cu
#  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
#)

# Collect all source files in this directory
file(GLOB_RECURSE SOURCE_FILES "*.h" "*.cuh" "*.cu" "*.cpp")
# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add target executable
OPTIX_add_executable(${TARGET_NAME} target_name ${SOURCE_FILES})
