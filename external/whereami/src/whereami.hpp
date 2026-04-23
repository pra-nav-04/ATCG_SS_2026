#pragma once

#include <string>

#include "whereami.h"

namespace wai {

/**
 * Returns the path to the current executable.
 *
 * Usage:
 *  - call `wai::getExecutablePath()` to retrieve the
 *    path
 *
 * @return the executable path on success, otherwise an empty string
 */
inline std::string getExecutablePath()
{
    int length = WAI_PREFIX(getExecutablePath)(nullptr, 0, nullptr);
    if (length <= 0)
        return {};

    std::string result;
    result.resize(length);
    length = WAI_PREFIX(getExecutablePath)(result.data(), length, nullptr);
    if (length <= 0)
        return {};
    return result;
}

WAI_FUNCSPEC
int WAI_PREFIX(getExecutablePath)(char* out, int capacity, int* dirname_length);

/**
 * Returns the path to the current module
 *
 * Usage:
 *  - call `wai::getModulePath()` to retrieve the path
 *
 * @return the module path on success, otherwise an empty string
 */
inline std::string getModulePath()
{
    int length = WAI_PREFIX(getModulePath)(nullptr, 0, nullptr);
    if (length <= 0)
        return {};

    std::string result;
    result.resize(length);
    length = WAI_PREFIX(getModulePath)(result.data(), length, nullptr);
    if (length <= 0)
        return {};
    return result;
}

} // namespace wai
