cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Auto-initialize CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 NEW)

# Add paths to our CMake code to the module path, so they can be found automatically by CMake.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Workaround for CMake not finding CUDA on Linux systems.
include(OpgCudaSetupHelpher)

project(OptiX-Playground LANGUAGES CUDA CXX C)

# Tells CMake to build all the libraries as shared libraries by default.
# This can be overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Enable position independent code also in non-shared libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Enable C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED)
# Consistently use the shared or static version of cudart.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Include CMake Macros.
# Custom wrapper around CUDA_WRAP_SRCS to pass different flags to PTX and OBJ compilations.
include(OpgCudaWrapPtxAndObjSrcs)

# Find at least a 11.0 version of CUDA.
find_package(CUDAToolkit 11.0 REQUIRED)

# Initialize arguments to the NVCC.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=\"--diag_suppress=2886\">
  )

# If your GPU is too old, hardcode architecture here:
# sm_50 corresponds to GTX 900 Series Mobile GPUs
# sm_52 corresponds to GTX 900 Desktop GPUs
#list(APPEND CUDA_NVCC_FLAGS -arch sm_50)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Enable M_PI definition
    add_compile_definitions(_USE_MATH_DEFINES)
    # Prevent windoof from defining `min` and `max` macros messing with std::min, std::max, etc.
    add_compile_definitions(NOMINMAX)

    # Disable "unknown pramga" warning.
    # We also need to tell nvcc that the compile option should be passed to the host compiler via -Xcompiler=...
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4068)

    # Compiler fails to write to .PDF file lately.... this is supposed to fix that.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /FS")
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/lib")
# Global output directory for compiled PTX modules
set(OPG_CUDA_GENERATED_OUTPUT_DIR_PTX "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ptx")

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/external/OptiX" CACHE PATH "Path to OptiX installed location.")

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories("${OptiX_INCLUDE}")

##################################################################
# Dependencies

# Third-party external libraries.
add_subdirectory(external)

# Core utility library.
add_subdirectory(opgutil)

#########################################################
# OPTIX_add_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
function(OPTIX_add_executable target_name_base target_name_var)

  set( target_name ${target_name_base} )
  set( ${target_name_var} ${target_name} PARENT_SCOPE )

  # Automatlically determine if .cu files should be compiled to ptx or obj, based on pragma defined in source file.
  OPG_CUDA_SOURCE_PROPERTY_FORMAT_FROM_PRAGMA(${ARGN})
  OPG_CUDA_SPLIT_PTX_SOURCE_FILES(REGULAR_SOURCE_FILES SOURCE_FILES_TO_PTX  ${ARGN})

  # Make the executable.
  add_executable(${target_name} ${REGULAR_SOURCE_FILES})

  # Compile the PTX files.
  if (SOURCE_FILES_TO_PTX)
    OPG_ADD_PTX_LIBRARY(${target_name} target_name_ptx ${SOURCE_FILES_TO_PTX})
    target_include_directories(${target_name_ptx} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target_name_ptx} PUBLIC opgutil glm)
  endif()

  # Pass the target name as compile-time constant
  target_compile_definitions( ${target_name} PRIVATE "OPG_TARGET_NAME_DEFINE=${target_name}" )

  # Link against the core utility library, that recursively depends on all other libraries that we might need.
  # NOTE: We build opgutil as a STATIC library.
  # To make sure that nothing that we might need is optimized away,
  # we need to tell the compiler that the whole archive should be included!
  #target_link_libraries( ${target_name} opgutil)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries( ${target_name}
      opgutil
      )
    set_target_properties(${target_name} PROPERTIES LINK_FLAGS "/WHOLEARCHIVE:opgutil")
  else()
    target_link_libraries( ${target_name}
      -Wl,--whole-archive
      opgutil
      -Wl,--no-whole-archive
      )
  endif()
endfunction()

#########################################################
#  List of exercises found in subdirectories.
#
add_subdirectory(src)
