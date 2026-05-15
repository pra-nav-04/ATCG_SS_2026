#include "whittedraygenerator.h"

#include "opg/scene/scene.h"
#include "opg/opg.h"
#include "opg/scene/components/camera.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

WhittedRayGenerator::WhittedRayGenerator(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    RayGenerator(std::move(_scene), _props)
{
    m_launch_params.alloc(1);
}

WhittedRayGenerator::~WhittedRayGenerator()
{
}

void WhittedRayGenerator::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    std::string ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "whittedraygenerator.cu");
    OptixProgramGroup progGroup = pipeline->addRaygenShader({ptx_filename, "__raygen__main"});
    OptixProgramGroup missProgGroup = pipeline->addMissShader({ptx_filename, "__miss__main"});
    OptixProgramGroup occlusionMissProgGroup = pipeline->addMissShader({ptx_filename, "__miss__occlusion"});

    sbt->addRaygenEntry(progGroup, nullptr);
    sbt->addMissEntry(missProgGroup, nullptr);
    sbt->addMissEntry(occlusionMissProgGroup, nullptr);
}

void WhittedRayGenerator::finalize()
{
    std::vector<const EmitterVPtrTable *> emitter_vptr_tables;
    m_scene->traverseSceneComponents<opg::Emitter>([&](opg::Emitter *emitter){
        emitter_vptr_tables.push_back(emitter->getEmitterVPtrTable());
    });

    m_emitters_buffer.alloc(emitter_vptr_tables.size());
    m_emitters_buffer.upload(emitter_vptr_tables.data());
}

void WhittedRayGenerator::launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer)
{
    // NOTE: We access tensors like numpy arrays.
    // 1st tensor dimension -> row -> y axis
    // 2nd tensor dimension -> column -> x axis
    uint32_t image_width = output_buffer.counts[1];
    uint32_t image_height = output_buffer.counts[0];

    WhittedLaunchParams launch_params;
    launch_params.scene_epsilon = 1e-3f;
    launch_params.output_radiance = output_buffer;
    launch_params.image_width = image_width;
    launch_params.image_height = image_height;

    launch_params.surface_interaction_trace_params.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.surface_interaction_trace_params.SBToffset = 0;
    launch_params.surface_interaction_trace_params.SBTstride = 1;
    launch_params.surface_interaction_trace_params.missSBTIndex = 0;

    launch_params.occlusion_trace_params.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    launch_params.occlusion_trace_params.SBToffset = 0;
    launch_params.occlusion_trace_params.SBTstride = 1;
    launch_params.occlusion_trace_params.missSBTIndex = 1;

    launch_params.traversable_handle = m_scene->getTraversableHandle(1);

    m_camera->getCameraData(launch_params.camera);

    launch_params.emitters = m_emitters_buffer.view();

    m_launch_params.upload(&launch_params);

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();
    OPTIX_CHECK( optixLaunch(pipeline->getPipeline(), stream, m_launch_params.getRaw(), m_launch_params.byteSize(), sbt->getSBT(), image_width, image_height, 1) );
    CUDA_SYNC_CHECK();
}

namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(WhittedRayGenerator, "raygen.whitted");

} // end namespace opg
