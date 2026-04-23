#
# Assembled from parts of FindCUDA.cmake
#

# Search for the cuda distribution.
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  # Search in the CUDA_BIN_PATH first.
  find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS
      ENV CUDA_TOOLKIT_ROOT
      ENV CUDA_PATH
      ENV CUDA_BIN_PATH
    PATH_SUFFIXES bin bin64
    DOC "CUDA Toolkit location."
    NO_DEFAULT_PATH
    )
  # Now search default paths
  find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS /opt/cuda/bin
          /usr/local/bin
          /usr/local/cuda/bin
    DOC "CUDA Toolkit location."
    )

  if (CUDA_TOOLKIT_ROOT_DIR)
    # Make sure that there is no /bin suffix in the path.
    string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
    # We need to force this back into the cache.
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "Toolkit location." FORCE)
  endif()

  if (NOT EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
      message(FATAL_ERROR "Specify CUDA_TOOLKIT_ROOT_DIR")
  endif ()
endif ()

# Search for the cuda compiler.
if(NOT CUDA_NVCC_EXECUTABLE)
  find_program(CUDA_NVCC_EXECUTABLE
    NAMES nvcc
    PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
    ENV CUDA_PATH
    ENV CUDA_BIN_PATH
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH
    )
  # Search default search paths, after we search our own set of paths.
  find_program(CUDA_NVCC_EXECUTABLE nvcc)
  mark_as_advanced(CUDA_NVCC_EXECUTABLE)
endif()

if( "${CMAKE_GENERATOR}" MATCHES "^Visual Studio")
  # When the old Visual Studio build system is used, we need to specify CMAKE_GENERATOR_TOOLSET
  SET(CMAKE_GENERATOR_TOOLSET "cuda=${CUDA_TOOLKIT_ROOT_DIR}")
else()
  # Otherwise, when the Ninja or GNU Make build system is used, we need to specify CMAKE_CUDA_COMPILER
  SET(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
endif()
