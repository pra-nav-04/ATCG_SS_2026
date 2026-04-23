#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <vector>
#include <string>

#include "opg/opgapi.h"


#define OPG_STRINGIFY2(name) #name
#define OPG_STRINGIFY(name) OPG_STRINGIFY2(name)
// OPG_TARGET_NAME resolves to the name of the cmake target (executable or library)
#define OPG_TARGET_NAME OPG_STRINGIFY(OPG_TARGET_NAME_DEFINE)

// The number of directory separators in the executable path relative to the project root directory
// This should be set in opgutil/CMakeLists.txt, but it defaults to 2 (${project_root}/bin/executable.exe)
#ifndef OPG_EXECUTABLE_SEPARATOR_COUNT
#define OPG_EXECUTABLE_SEPARATOR_COUNT 2
#endif

namespace opg {

OPG_API std::vector<std::string> splitString(const std::string &input, const std::string &delim = " ", bool allow_empty = false);

// Read complete content of a file.
OPG_API std::vector<char> readFile(const char *filename);

// Get *root* directory of the CMake project independent of current working directory, relative to executable location
OPG_API std::string getRootPath();

// Construct the absolute path to the ptx file that was compiled from `sourceName` in the CMake target `targetName`.
OPG_API std::string getPtxFilename(const char *targetName, const char *sourceName);

// Create a context for calling into the OptiX API.
OPG_API OptixDeviceContext createOptixContext(CUcontext cuCtx = nullptr);

} // end namespace opg
