#pragma once

#include <optix.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <array>

#include "opg/opgapi.h"

struct OptixProgramGroupEntryKey
{
    OptixModule module              = nullptr;
    std::string entryname;

    OptixProgramGroupEntryKey() = default;
    OptixProgramGroupEntryKey(OptixModule _module, const char *_entryname) :
        module { _module },
        entryname { _entryname != nullptr ? _entryname : "" }
    {}

    bool operator == (const OptixProgramGroupEntryKey &other) const
    {
        return module == other.module &&
               entryname == other.entryname;
    }
};

struct OptixProgramGroupDescKey
{
    OptixProgramGroupKind kind;
    unsigned int flags          = 0;

    std::array<OptixProgramGroupEntryKey, 3> entryPoints;

    OptixProgramGroupDescKey() = default;
    OptixProgramGroupDescKey(const OptixProgramGroupDesc &desc) :
        kind { desc.kind },
        flags { desc.flags }
    {
        switch(kind)
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                entryPoints[0] = { desc.raygen.module, desc.raygen.entryFunctionName };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_MISS:
                entryPoints[0] = { desc.miss.module, desc.miss.entryFunctionName };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
                entryPoints[0] = { desc.exception.module, desc.exception.entryFunctionName };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                entryPoints[0] = { desc.hitgroup.moduleCH, desc.hitgroup.entryFunctionNameCH };
                entryPoints[1] = { desc.hitgroup.moduleAH, desc.hitgroup.entryFunctionNameAH };
                entryPoints[2] = { desc.hitgroup.moduleIS, desc.hitgroup.entryFunctionNameIS };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
                entryPoints[0] = { desc.callables.moduleDC, desc.callables.entryFunctionNameDC };
                entryPoints[1] = { desc.callables.moduleCC, desc.callables.entryFunctionNameCC };
                break;
        }
    }

    bool operator == (const OptixProgramGroupDescKey &other) const
    {
        return kind == other.kind &&
            flags == other.flags &&
            entryPoints[0] == other.entryPoints[0] &&
            entryPoints[1] == other.entryPoints[1] &&
            entryPoints[2] == other.entryPoints[2];
    }

};


template <typename T>
inline void hash_combine(std::size_t &s, const T &v)
{
    std::hash<T> h;
    s^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

namespace std {

    template<>
    struct hash<OptixProgramGroupEntryKey>
    {
        size_t operator()(const OptixProgramGroupEntryKey &v) const
        {
            size_t s = 0;
            hash_combine(s, v.module);
            hash_combine(s, v.entryname);
            return s;
        }
    };

    template<>
    struct hash<OptixProgramGroupDescKey>
    {
        size_t operator()(const OptixProgramGroupDescKey &v) const
        {
            size_t s = 0;
            hash_combine(s, v.kind);
            hash_combine(s, v.flags);
            for (const auto &e : v.entryPoints)
            {
                hash_combine(s, e);
            }
            return s;
        }
    };

} // end namespace std


namespace opg {

struct ShaderEntryPointDesc
{
    std::string ptx_filename;
    std::string entrypoint_name;
};

class RayTracingPipeline
{
public:
    OPG_API RayTracingPipeline(OptixDeviceContext context);
    OPG_API ~RayTracingPipeline();

    OPG_API OptixProgramGroup addRaygenShader(const ShaderEntryPointDesc &raygen_shader_desc);
    OPG_API OptixProgramGroup addCallableShader(const ShaderEntryPointDesc &callable_shader_desc);
    OPG_API OptixProgramGroup addMissShader(const ShaderEntryPointDesc &miss_shader_desc);
    OPG_API OptixProgramGroup addTrianglesHitGroupShader(const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc);
    OPG_API OptixProgramGroup addProceduralHitGroupShader(const ShaderEntryPointDesc &intersection_shader_desc, const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc);

    OPG_API void createPipeline();

    inline OptixPipeline getPipeline() const { return m_pipeline; }

private:
    OptixModule createNewModule(const std::string &ptx_filename);
    OptixModule getCachedModule(const std::string &ptx_filename);

    OptixProgramGroup createNewProgramGroup(const OptixProgramGroupDesc &prog_group_desc);
    OptixProgramGroup getCachedProgramGroup(const OptixProgramGroupDesc &prog_group_desc);

private:
    OptixDeviceContext          m_context;

    std::unordered_map<std::string, OptixModule> m_module_cache;
    std::unordered_map<OptixProgramGroupDescKey, OptixProgramGroup> m_program_groups;

    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixModuleCompileOptions   m_module_compile_options   = {};
    OptixProgramGroupOptions    m_program_group_options    = {};
    OptixPipelineLinkOptions    m_pipeline_link_options    = {};

    OptixPipeline               m_pipeline                 = 0;
};

} // end namespace opg
