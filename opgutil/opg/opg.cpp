#include "opg/opg.h"

#include "opg/exception.h"

#include "opg/raytracing/opg_optix_stubs.h"
// This file must be included in exactly one cpp file!
#include <optix_function_table_definition.h>

#include <whereami.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <fmt/core.h>

namespace opg {

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

OptixDeviceContext createOptixContext(CUcontext cuCtx)
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( nullptr ) );

    OPTIX_CHECK( optixInit() );

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;

    OptixDeviceContext context;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    return context;
}

std::vector<std::string> splitString(const std::string &input, const std::string &delim, bool allow_empty)
{
    std::vector<std::string> result;
    auto begin_partial = input.begin();
    for (auto it = input.begin(); it != input.end(); ++it)
    {
        if (delim.find(*it) != std::string::npos)
        {
            if (allow_empty || begin_partial != it)
                result.emplace_back(begin_partial, it);
            begin_partial = std::next(it);
        }
    }
    if (allow_empty || begin_partial != input.end())
        result.emplace_back(begin_partial, input.end());
    return result;
}


std::vector<char> readFile(const char *filename)
{
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open())
    {
        throw std::runtime_error(fmt::format("Could not open file '{}'!", filename));
    }

    input.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    input.read(buffer.data(), buffer.size());

    return buffer;
}

std::string getRootPath()
{
    std::string executablePath = wai::getExecutablePath();
    // Turn backslashes into forward slashes
    for (auto &c : executablePath)
        if (c == '\\') c = '/';
    // Cutoff filename and add path suffix
    size_t endIndex = executablePath.size();
    // executablePath = ".../root_dir/bin/abc.exe"
    // Find second-last separator to split executable path into ".../root_dir" and "/bin/abc.exe"
    for (int i = 0; i < OPG_EXECUTABLE_SEPARATOR_COUNT; ++i)
    {
        endIndex = executablePath.find_last_of('/', endIndex-1);
        if (endIndex == 0 || endIndex == std::string::npos)
            throw std::runtime_error("getPtxFilename: Failed to extract directory from executable path!");
    }
    std::string rootPath = executablePath.substr(0, endIndex);
    return rootPath;
}

std::string getPtxFilename(const char *targetName, const char *sourceName)
{
    std::ostringstream ss;

    // Skip directories in sourceName
    size_t begin = 0;
    size_t end = 0;
    size_t i;
    for (i = 0; sourceName[i] != '\0'; ++i)
    {
        if (sourceName[i] == '/')
            begin = i+1; // 1 beyond separator
        if (sourceName[i] == '.')
            end = i; // Exclude .
    }
    if (end <= begin)
        end = i;

    ss << getRootPath() << "/lib/ptx/" << targetName << "/"; // << sourceName << ".ptx";
    ss.write(sourceName+begin, end-begin);
    ss << ".ptx";
    return ss.str();
}

} // end namespace opg
