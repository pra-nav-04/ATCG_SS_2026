#include "radiosityraygenerator.h"
#include "radiositykernels.h"

#include "opg/scene/scene.h"
#include "opg/opg.h"
#include "opg/scene/components/camera.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>


RadiosityRayGenerator::RadiosityRayGenerator(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    RayGenerator(std::move(_scene), _props)
{
    m_launch_params.alloc(1);
}

RadiosityRayGenerator::~RadiosityRayGenerator()
{
}

void RadiosityRayGenerator::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    std::string ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "radiosityraygenerator.cu");
    OptixProgramGroup generateRaygenProgGroup = pipeline->addRaygenShader({ptx_filename, "__raygen__generateRadiosity"});
    OptixProgramGroup renderRaygenProgGroup = pipeline->addRaygenShader({ptx_filename, "__raygen__renderRadiosity"});
    OptixProgramGroup surfaceMissProgGroup = pipeline->addMissShader({ptx_filename, "__miss__main"});
    OptixProgramGroup occlusionMissProgGroup = pipeline->addMissShader({ptx_filename, "__miss__occlusion"});

    m_generateRadiosityRaygenIndex  = sbt->addRaygenEntry(generateRaygenProgGroup, nullptr);
    m_renderRadiosityRaygenIndex    = sbt->addRaygenEntry(renderRaygenProgGroup, nullptr);
    m_surfaceMissIndex              = sbt->addMissEntry(surfaceMissProgGroup, nullptr);
    m_occlusionMissIndex            = sbt->addMissEntry(occlusionMissProgGroup, nullptr);
}

void RadiosityRayGenerator::finalize()
{
    // Collect all surfaces (emitters) participating in the radiosity method
    m_radiosityEmitters.clear();
    m_total_primitive_count = 0;
    m_scene->traverseSceneComponents<RadiosityEmitter>([&](RadiosityEmitter *emitter){
        std::cout << "emitter with primitives " << emitter->m_primitiveCount << std::endl;
        m_total_primitive_count += emitter->m_primitiveCount;
        m_radiosityEmitters.push_back(emitter);
    });

    m_form_factor_matrix_buffer.alloc(m_total_primitive_count * m_total_primitive_count);

    // TODO do this in launch frame?!
    computeFormFactor();
    computeRadiosity();
}

void RadiosityRayGenerator::launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer)
{
    // NOTE: We access tensors like numpy arrays.
    // 1st tensor dimension -> row -> y axis
    // 2nd tensor dimension -> column -> x axis
    uint32_t image_width = output_buffer.counts[1];
    uint32_t image_height = output_buffer.counts[0];

    RadiosityLaunchParams launch_params;
    launch_params.scene_epsilon = 1e-3f;
    launch_params.output_radiance = output_buffer;
    launch_params.image_width = image_width;
    launch_params.image_height = image_height;

    m_camera->getCameraData(launch_params.camera);

    launch_params.surface_interaction_trace_params.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.surface_interaction_trace_params.SBToffset = 0;
    launch_params.surface_interaction_trace_params.SBTstride = 1;
    launch_params.surface_interaction_trace_params.missSBTIndex = 0;

    launch_params.occlusion_trace_params.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    launch_params.occlusion_trace_params.SBToffset = 0;
    launch_params.occlusion_trace_params.SBTstride = 1;
    launch_params.occlusion_trace_params.missSBTIndex = 1;

    launch_params.traversable_handle = m_scene->getTraversableHandle(1);

    m_launch_params.upload(&launch_params);

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();
    OPTIX_CHECK( optixLaunch(pipeline->getPipeline(), stream, m_launch_params.getRaw(), m_launch_params.byteSize(), sbt->getSBT(m_renderRadiosityRaygenIndex), image_width, image_height, 1) );
    CUDA_SYNC_CHECK();
}


void RadiosityRayGenerator::computeFormFactor()
{
    cudaStream_t stream = nullptr; // TODO

    auto pipeline = m_scene->getRayTracingPipeline();
    auto sbt = m_scene->getSBT();

    RadiosityLaunchParams launch_params;
    launch_params.scene_epsilon = 1e-3f;

    launch_params.surface_interaction_trace_params.rayFlags = OPTIX_RAY_FLAG_NONE;
    launch_params.surface_interaction_trace_params.SBToffset = 0;
    launch_params.surface_interaction_trace_params.SBTstride = 1;
    launch_params.surface_interaction_trace_params.missSBTIndex = 0;

    launch_params.occlusion_trace_params.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    launch_params.occlusion_trace_params.SBToffset = 0;
    launch_params.occlusion_trace_params.SBTstride = 1;
    launch_params.occlusion_trace_params.missSBTIndex = 1;

    launch_params.traversable_handle = m_scene->getTraversableHandle(1);

    launch_params.form_factor_matrix = m_form_factor_matrix_buffer.data();
    launch_params.form_factor_matrix_size = m_total_primitive_count;

    size_t primitive_offset_i = 0;
    for ( size_t i = 0; i < m_radiosityEmitters.size(); ++i )
    {
        const RadiosityEmitter *radiosity_emitter_i = m_radiosityEmitters[i];
        const MeshShapeData &mesh_data_i = radiosity_emitter_i->getMeshShape()->getMeshShapeData();
        size_t primitive_count_i = radiosity_emitter_i->m_primitiveCount;

        size_t primitive_offset_j = 0;
        for ( size_t j = 0; j <= i; ++j )
        {
            const RadiosityEmitter *radiosity_emitter_j = m_radiosityEmitters[j];
            const MeshShapeData &mesh_data_j = radiosity_emitter_j->getMeshShape()->getMeshShapeData();
            size_t primitive_count_j = radiosity_emitter_j->m_primitiveCount;

            // Compute radiosity between meshes i and j
            launch_params.instance_1.indices = mesh_data_i.indices;
            launch_params.instance_1.positions = mesh_data_i.positions;
            launch_params.instance_1.normals = mesh_data_i.normals;
            launch_params.instance_1.transform = radiosity_emitter_i->getShapeInstance()->getTransform();
            launch_params.instance_1.form_factor_matrix_offset = primitive_offset_i;

            launch_params.instance_2.indices = mesh_data_j.indices;
            launch_params.instance_2.positions = mesh_data_j.positions;
            launch_params.instance_2.normals = mesh_data_j.normals;
            launch_params.instance_2.transform = radiosity_emitter_j->getShapeInstance()->getTransform();
            launch_params.instance_2.form_factor_matrix_offset = primitive_offset_j;

            std::cout << "Launch " << i << " <-> " << j << std::endl;

            // Launch

            m_launch_params.upload(&launch_params);

            OPTIX_CHECK( optixLaunch(
                pipeline->getPipeline(),
                stream,
                m_launch_params.getRaw(),
                m_launch_params.byteSize(),
                sbt->getSBT(m_generateRadiosityRaygenIndex),
                primitive_count_i,  // launch width
                primitive_count_j,  // launch height
                1                   // launch depth
            ) );
            CUDA_SYNC_CHECK();

            std::cout << "Finish " << i << " <-> " << j << std::endl;

            primitive_offset_j += primitive_count_j;
        }
        primitive_offset_i += primitive_count_i;
    }
}

void RadiosityRayGenerator::computeRadiosity()
{
    // This function solves K * radiosity = emission for radiosity
    // for K := identity - diag(albedo) * form_factor_matrix

    // Build the emissions vector in host memory
    std::vector<glm::vec3> emissions(m_total_primitive_count);
    std::vector<glm::vec3> albedos(m_total_primitive_count);
    size_t primitive_offset = 0;
    for (const auto &radiosity_emitter : m_radiosityEmitters)
    {
        size_t primitive_count = radiosity_emitter->m_primitiveCount;
        // Fill emission
        glm::vec3 emission = radiosity_emitter->m_emission;
        std::fill(emissions.begin() + primitive_offset, emissions.begin() + primitive_offset + primitive_count, emission);
        // Fill albedo
        glm::vec3 albedo = radiosity_emitter->m_albedo;
        std::fill(albedos.begin() + primitive_offset, albedos.begin() + primitive_offset + primitive_count, albedo);
        // Advance primitive offset
        primitive_offset += primitive_count;
    }

    opg::DeviceBuffer<glm::vec3> emissions_buffer;
    emissions_buffer.alloc(m_total_primitive_count);
    emissions_buffer.upload(emissions.data());

    opg::DeviceBuffer<glm::vec3> albedos_buffer;
    albedos_buffer.alloc(m_total_primitive_count);
    albedos_buffer.upload(albedos.data());

    float lambda = 1.0f;
    const float convergence_threshold = 1e-4f;
    const uint32_t max_iterations = 256;

    /* Implement:
     * - Initialize the radiosity solution with the emission
     * - Compute the radiosity iteratively using the Jacobi method: radiosity = radiosity + lambda * (emissions - K * radiosity)
     * - Write the resulting radiosity into the `RadiosityEmitter::m_primitiveRadiosity` buffer for each emitter in `m_radiosityEmitters`.
     * - Hint: you can write your own CUDA kernels in radiositykernels.h/cu.
     */
    opg::DeviceBuffer<glm::vec3> previous_radiosity_buffer;
    previous_radiosity_buffer.alloc(m_total_primitive_count);
    initializeRadiosityBuffer(previous_radiosity_buffer.view(), emissions_buffer.view());

    opg::DeviceBuffer<glm::vec3> next_radiosity_buffer;
    next_radiosity_buffer.alloc(m_total_primitive_count);

    opg::DeviceBuffer<float> deltas_buffer;
    deltas_buffer.alloc(m_total_primitive_count);
    std::vector<float> deltas_host(m_total_primitive_count, 0.0f);

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration)
    {
        jacobiRadiosityStep(
            m_form_factor_matrix_buffer.view(),
            static_cast<uint32_t>(m_total_primitive_count),
            albedos_buffer.view(),
            emissions_buffer.view(),
            previous_radiosity_buffer.view(),
            next_radiosity_buffer.view(),
            deltas_buffer.view(),
            lambda);

        deltas_buffer.download(deltas_host.data());
        const float max_delta = *std::max_element(deltas_host.begin(), deltas_host.end());

        previous_radiosity_buffer = std::move(next_radiosity_buffer);
        next_radiosity_buffer.alloc(m_total_primitive_count);

        if (max_delta < convergence_threshold)
        {
            std::cout << "Radiosity converged after " << (iteration + 1)
                      << " iterations with max delta " << max_delta << std::endl;
            break;
        }
    }

    primitive_offset = 0;
    for (const auto &radiosity_emitter : m_radiosityEmitters)
    {
        const size_t primitive_count = radiosity_emitter->m_primitiveCount;
        CUDA_CHECK(cudaMemcpy(
            radiosity_emitter->m_primitiveRadiosity.data(),
            previous_radiosity_buffer.data(primitive_offset),
            primitive_count * sizeof(glm::vec3),
            cudaMemcpyDeviceToDevice));
        primitive_offset += primitive_count;
    }
}


namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(RadiosityRayGenerator, "raygen.radiosity");

} // end namespace opg
