#include "opg/scene/components/shapes/torus.h"
#include "opg/scene/interface/bsdf.h"
#include "opg/scene/interface/emitter.h"
#include "opg/scene/scene.h"

#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/exception.h"
#include "opg/opg.h"

#include "opg/scene/sceneloader.h"

#include <vector>
#include "opg/raytracing/opg_optix_stubs.h"

namespace opg {

TorusShape::TorusShape(PrivatePtr<Scene> _scene, const Properties &_props) :
    Shape(std::move(_scene), _props)
{
    m_data.minorRadius = _props.getFloat("minor_radius", 0.2f);
    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);

    m_data_ptr = m_data_buffer.data();
}

TorusShape::~TorusShape()
{
}

void TorusShape::prepareAccelerationStructure()
{
    uint32_t geometry_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE; // OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    GenericDeviceBuffer d_temp;
    DeviceBuffer<OptixAabb> d_aabb;

    std::vector<CUdeviceptr> build_input_aabbBuffers;

    OptixAabb aabb;
    aabb.minX = -2*m_data.minorRadius-1;
    aabb.minY = -2*m_data.minorRadius-1;
    aabb.minZ = -2*m_data.minorRadius;
    aabb.maxX = 2*m_data.minorRadius+1;
    aabb.maxY = 2*m_data.minorRadius+1;
    aabb.maxZ = 2*m_data.minorRadius;
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

void TorusShape::initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt)
{
    auto ptx_filename = getPtxFilename(OPG_TARGET_NAME, "opg/scene/components/shapes/torus.cu");
    m_hit_prog_group = pipeline->addProceduralHitGroupShader({ ptx_filename, "__intersection__torus" }, { ptx_filename, "__closesthit__torus" }, {});

    m_sample_position_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__torus_sample_position" });
    m_eval_position_sampling_pdf_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__torus_eval_position_sampling_pdf" });
    m_sample_next_event_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__torus_sample_next_event" });
    m_eval_next_event_sampling_pdf_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__torus_eval_next_event_sampling_pdf" });
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY(TorusShape, "shape.torus");

} // end namespace opg
