#include "opg/scene/components/shapes/sphere.h"

#include "opg/raytracing/raytracingpipeline.h"
#include "opg/opg.h"
#include "opg/exception.h"
#include "opg/scene/scene.h"
#include "opg/scene/interface/bsdf.h"
#include "opg/scene/interface/emitter.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

namespace opg {

SphereShape::SphereShape(PrivatePtr<Scene> _scene, const Properties &_props) :
    Shape(std::move(_scene), _props)
{
}

SphereShape::~SphereShape()
{
}

void SphereShape::prepareAccelerationStructure()
{
    uint32_t geometry_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE; // OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    GenericDeviceBuffer d_temp;
    DeviceBuffer<OptixAabb> d_aabb;

    std::vector<CUdeviceptr> build_input_aabbBuffers;

    OptixAabb aabb;
    aabb.minX = -1;
    aabb.minY = -1;
    aabb.minZ = -1;
    aabb.maxX = 1;
    aabb.maxY = 1;
    aabb.maxZ = 1;
    d_aabb.allocIfRequired(1);
    d_aabb.upload(&aabb);

    build_input_aabbBuffers.resize(1);
    build_input_aabbBuffers[0] = d_aabb.getRaw();

    OptixBuildInput build_input = {};
    build_input.type                           = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers          = build_input_aabbBuffers.data();
    build_input.customPrimitiveArray.numPrimitives        = 1;
    build_input.customPrimitiveArray.flags                = &geometry_input_flags;
    build_input.customPrimitiveArray.numSbtRecords        = 1;
    build_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(m_scene->getContext(), &accel_options, &build_input, 1, &gas_buffer_sizes) );

    d_temp.allocIfRequired(gas_buffer_sizes.tempSizeInBytes);

    m_ast_buffer.alloc(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK( optixAccelBuild(
        m_scene->getContext(), 0,   // CUDA stream
        &accel_options,
        &build_input,
        1,
        d_temp.getRaw(),
        d_temp.byteSize(),
        m_ast_buffer.getRaw(),
        m_ast_buffer.byteSize(),
        &m_ast_handle,
        nullptr,
        0
    ) );
}

void SphereShape::initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt)
{
    // TODO this probably also makes sense on a per-instance basis, such that the anyhit can be influenced by the BSDF?
    auto ptx_filename = getPtxFilename(OPG_TARGET_NAME, "opg/scene/components/shapes/sphere.cu");
    m_hit_prog_group = pipeline->addProceduralHitGroupShader({ ptx_filename, "__intersection__sphere" }, { ptx_filename, "__closesthit__sphere" }, {});

    m_sample_position_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__sphere_sample_position" });
    m_eval_position_sampling_pdf_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__sphere_eval_position_sampling_pdf" });
    m_sample_next_event_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__sphere_sample_next_event" });
    m_eval_next_event_sampling_pdf_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__sphere_eval_next_event_sampling_pdf" });
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY(SphereShape, "shape.sphere");

} // end namespace opg
