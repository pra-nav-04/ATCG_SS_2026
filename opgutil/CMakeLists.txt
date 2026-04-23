
file(GLOB_RECURSE SOURCE_FILES "*.h" "*.cpp" "*.cu")

set(opg_target "opgutil")

# Automatlically determine if .cu files should be compiled to ptx or obj, based on pragma defined in source file.
OPG_CUDA_SOURCE_PROPERTY_FORMAT_FROM_PRAGMA(${SOURCE_FILES})
OPG_CUDA_SPLIT_PTX_SOURCE_FILES(REGULAR_SOURCE_FILES SOURCE_FILES_TO_PTX  ${SOURCE_FILES})

# Make the library.
add_library(${opg_target} STATIC ${REGULAR_SOURCE_FILES})

# Compile the PTX files.
if (SOURCE_FILES_TO_PTX)
  OPG_ADD_PTX_LIBRARY(${opg_target} opg_target_ptx ${SOURCE_FILES_TO_PTX})
  target_include_directories(${opg_target_ptx} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${opg_target_ptx} PUBLIC glm)
endif()

# Pass the target name as compile-time constant
target_compile_definitions( ${opg_target} PRIVATE "OPG_TARGET_NAME_DEFINE=${opg_target}" )

# If a regular Visual Studio project was created with the standalone cmake version on Windows:
# executablePath = ".../root_dir/bin/Debug/abc.exe"
# Then we need to find the third-last separator to split executable path into ".../root_dir" and "bin/Debug/abc.exe"
# If the project is opened directly in Visual Studio, the executablePath is the same as on other OSes.
# executablePath = ".../root_dir/bin/abc.exe"
#if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
if( "${CMAKE_GENERATOR}" MATCHES "^Visual Studio")
  target_compile_definitions( ${opg_target} PUBLIC "OPG_EXECUTABLE_SEPARATOR_COUNT=3")
endif()

target_include_directories(${opg_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${opg_target}
  PUBLIC
    fmt::fmt
    glm
    pugixml
    glfw
    glad
    CUDA::cudart
    libnpy
  PRIVATE
    imgui
    nfd
    tinyexr
    tinyobj
    stb
    whereami
)
