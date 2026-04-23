#include "opg/raytracing/raytracingpipeline.h"
#include "opg/opg.h"
#include "opg/exception.h"

#include "opg/raytracing/opg_optix_stubs.h"

#include <algorithm>

namespace opg {

RayTracingPipeline::RayTracingPipeline(OptixDeviceContext context) :
    m_context { context }
{
    // Set default module compile options
    m_module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    // Set default pipeline compile options
    m_pipeline_compile_options.usesMotionBlur        = false;
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; // OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues      = 3; // radiance uses 3, occlusion uses 1
    m_pipeline_compile_options.numAttributeValues    = 2; // two values for barycentric coordinates on a triangle
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    // Set default pipeline link options
    m_pipeline_link_options.maxTraceDepth            = 2;//MAX_TRACE_DEPTH;
#if OPTIX_VERSION < 70700
    m_pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
}

RayTracingPipeline::~RayTracingPipeline()
{
    optixPipelineDestroy(m_pipeline);

    for (const auto & [ key, prog_group ] : m_program_groups)
    {
        optixProgramGroupDestroy(prog_group);
    }

    for (const auto & [ filename, module ] : m_module_cache)
    {
        optixModuleDestroy(module);
    }
}

OptixProgramGroup RayTracingPipeline::createNewProgramGroup(const OptixProgramGroupDesc &prog_group_desc)
{
    // TODO this needs to be cached?!?!
    // Yes, this is indeed the correct level for caching..
    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    OptixProgramGroup prog_group;
    OPTIX_CHECK_LOG( optixProgramGroupCreate( m_context, &prog_group_desc,
                                            1,  // num program groups
                                            &m_program_group_options, log, &sizeof_log, &prog_group ) );

    return prog_group;
}

OptixProgramGroup RayTracingPipeline::getCachedProgramGroup(const OptixProgramGroupDesc &prog_group_desc)
{
    OptixProgramGroup &prog_group = m_program_groups[prog_group_desc];
    if (prog_group == nullptr)
        prog_group = createNewProgramGroup(prog_group_desc);
    return prog_group;
}

OptixProgramGroup RayTracingPipeline::addRaygenShader(const ShaderEntryPointDesc &raygen_shader_desc)
{
    OptixModule ptx_module = getCachedModule(raygen_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc    = {};
    prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    prog_group_desc.raygen.module            = ptx_module;
    prog_group_desc.raygen.entryFunctionName = raygen_shader_desc.entrypoint_name.c_str();

    return getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addCallableShader(const ShaderEntryPointDesc &callable_shader_desc)
{
    OptixModule ptx_module = getCachedModule(callable_shader_desc.ptx_filename);

    // TODO directs VS continuation callable!!!

    OptixProgramGroupDesc prog_group_desc         = {};
    prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

    if (callable_shader_desc.entrypoint_name.rfind("__direct_callable__", 0) == 0)
    {
        prog_group_desc.callables.moduleDC            = ptx_module;
        prog_group_desc.callables.entryFunctionNameDC = callable_shader_desc.entrypoint_name.c_str();
    }
    else if (callable_shader_desc.entrypoint_name.rfind("__continuation_callable__", 0) == 0)
    {
        prog_group_desc.callables.moduleCC            = ptx_module;
        prog_group_desc.callables.entryFunctionNameCC = callable_shader_desc.entrypoint_name.c_str();
    }
    else
    {
        std::stringstream ss;
        ss << "Not a callable program entry point: " << callable_shader_desc.entrypoint_name;
        throw std::runtime_error(ss.str());
    }

    return getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addMissShader(const ShaderEntryPointDesc &miss_shader_desc)
{
    OptixModule ptx_module = getCachedModule(miss_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc  = {};
    prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    prog_group_desc.miss.module            = ptx_module;
    prog_group_desc.miss.entryFunctionName = miss_shader_desc.entrypoint_name.c_str();

    return getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addTrianglesHitGroupShader(const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc)
{
    OptixModule ptx_module_ch = getCachedModule(closestHit_shader_desc.ptx_filename);
    OptixModule ptx_module_ah = getCachedModule(anyHit_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc        = {};
    prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.hitgroup.moduleCH            = ptx_module_ch;
    prog_group_desc.hitgroup.entryFunctionNameCH = ptx_module_ch ? closestHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleAH            = ptx_module_ah;
    prog_group_desc.hitgroup.entryFunctionNameAH = ptx_module_ah ? anyHit_shader_desc.entrypoint_name.c_str() : nullptr;

    return getCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup RayTracingPipeline::addProceduralHitGroupShader(const ShaderEntryPointDesc &intersection_shader_desc, const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc)
{
    OptixModule ptx_module_ch = getCachedModule(closestHit_shader_desc.ptx_filename);
    OptixModule ptx_module_ah = getCachedModule(anyHit_shader_desc.ptx_filename);
    OptixModule ptx_module_is = getCachedModule(intersection_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc        = {};
    prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.hitgroup.moduleCH            = ptx_module_ch;
    prog_group_desc.hitgroup.entryFunctionNameCH = ptx_module_ch ? closestHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleAH            = ptx_module_ah;
    prog_group_desc.hitgroup.entryFunctionNameAH = ptx_module_ah ? anyHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleIS            = ptx_module_is;
    prog_group_desc.hitgroup.entryFunctionNameIS = ptx_module_is ? intersection_shader_desc.entrypoint_name.c_str() : nullptr;

    return getCachedProgramGroup(prog_group_desc);
}

OptixModule RayTracingPipeline::createNewModule(const std::string &ptx_filename)
{
    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    if (ptx_filename.empty())
    {
        return nullptr;
    }

    OptixModule ptx_module;

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif
    std::vector<char> ptxSource = opg::readFile(ptx_filename.c_str());
    OPTIX_CHECK_LOG( optixModuleCreate(
        m_context,
        &m_module_compile_options,
        &m_pipeline_compile_options,
        ptxSource.data(),
        ptxSource.size(),
        log,
        &sizeof_log,
        &ptx_module
    ) );
#if OPTIX_VERSION < 70700
#undef optixModuleCreate
#endif

    return ptx_module;
}

OptixModule RayTracingPipeline::getCachedModule(const std::string &ptx_filename)
{
    OptixModule &ptx_module = m_module_cache[ptx_filename];
    if (ptx_module == nullptr)
        ptx_module = createNewModule(ptx_filename);
    return ptx_module;
}

void RayTracingPipeline::createPipeline()
{
    char   log[2048];
    size_t sizeof_log = sizeof(log);

    // Collect all program groups in plain vector
    std::vector<OptixProgramGroup> program_groups;
    std::transform(m_program_groups.begin(), m_program_groups.end(), std::back_inserter(program_groups), [](const auto &entry){return entry.second;});

    OPTIX_CHECK_LOG( optixPipelineCreate(m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                        program_groups.data(), program_groups.size(), log,
                                        &sizeof_log, &m_pipeline) );
}

} // end namespace opg
