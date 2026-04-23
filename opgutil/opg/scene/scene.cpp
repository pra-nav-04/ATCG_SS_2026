#include "opg/scene/scene.h"

#include "opg/raytracing/opg_optix_stubs.h"

namespace opg {

Scene::Scene(OptixDeviceContext context) :
    m_context { context },
    m_pipeline { context },
    m_sbt { context }
{
}

Scene::~Scene()
{
}

SceneComponent *Scene::getSceneComponentByName(const std::string &name)
{
    auto it = m_namedComponents.find(name);
    if (it == m_namedComponents.end())
        return nullptr;
    return it->second;
}

void Scene::addComponent(std::unique_ptr<SceneComponent> component)
{
    SceneComponent *rawComponent = component.get();

    if (!rawComponent->getName().empty())
    {
        m_namedComponents[rawComponent->getName()] = rawComponent;
    }

    Shape* shape = dynamic_cast<Shape*>(rawComponent);
    if (shape != nullptr)
    {
        m_shapes.push_back(shape);
    }
    ShapeInstance* shapeInstance = dynamic_cast<ShapeInstance*>(rawComponent);
    if (shapeInstance != nullptr)
    {
        m_shapeInstances.push_back(shapeInstance);
    }

    /*
    Emitter* emitter = dynamic_cast<Emitter*>(rawComponent);
    if (emitter != nullptr)
    {
        m_emitters.push_back(emitter);
    }
    Sensor* sensor = dynamic_cast<Sensor*>(rawComponent);
    if (sensor != nullptr)
    {
        if (m_sensor != nullptr)
        {
            throw std::runtime_error("Only one sensor supported per scene right now!");
        }
        m_sensor = sensor;
    }
    */

    RayGenerator *rayGenerator = dynamic_cast<RayGenerator*>(rawComponent);
    if (rayGenerator != nullptr)
    {
        if (m_rayGenerator != nullptr)
        {
            throw std::runtime_error("Only one ray generator supported per scene right now!");
        }
        m_rayGenerator = rayGenerator;
    }

    m_components.push_back(std::move(component));
}

void Scene::buildIAS(uint32_t rayCount)
{
    const size_t num_instances = m_shapeInstances.size();

    std::vector<OptixInstance> optix_instances(num_instances);

    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < m_shapeInstances.size(); ++i )
    {
        auto& shape_instance = m_shapeInstances[i];
        auto& optix_instance = optix_instances[i];
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<uint32_t>(i); // TODO
        optix_instance.sbtOffset         = sbt_offset;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = shape_instance->getShape()->getAST();

        // shape_instance->getTransform() is a 4x4 matrix in column-major layout
        // glm::mat4x3 is 4 columns, 3 rows, column-major matrix
        // optix_instance.transform is a 3 rows, 4 columns, row-major matrix
        reinterpret_cast<glm::mat3x4&>(optix_instance.transform) = glm::transpose(glm::mat4x3(shape_instance->getTransform()));

        sbt_offset += rayCount;  // one sbt record per GAS build input per RAY_TYPE
    }

    DeviceBuffer<OptixInstance> d_instances(num_instances);
    d_instances.upload(optix_instances.data());

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances.getRaw();
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        m_context,
        &accel_options,
        &instance_input,
        1, // num build inputs
        &ias_buffer_sizes
    ) );

    GenericDeviceBuffer d_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    m_ias_buffer.alloc(ias_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK( optixAccelBuild(
        m_context,
        nullptr,                  // CUDA stream
        &accel_options,
        &instance_input,
        1,                  // num build inputs
        d_temp_buffer.getRaw(),
        ias_buffer_sizes.tempSizeInBytes,
        m_ias_buffer.getRaw(),
        ias_buffer_sizes.outputSizeInBytes,
        &m_ias_handle,
        nullptr,            // emitted property list
        0                   // num emitted properties
    ) );
}


void Scene::finalize()
{
    for (auto &shape : m_shapes)
    {
        shape->prepareAccelerationStructure();
    }
    buildIAS(1);

    for (auto &component : m_components)
    {
        component->ensurePipelineInitialized(&m_pipeline, &m_sbt);
    }
    m_pipeline.createPipeline();
    m_sbt.createSBT();

    for (auto &component : m_components)
    {
        component->finalize();
    }
}

void Scene::traceRays(const TensorView<glm::vec3, 2> &output_buffer)
{
    /// TODO notify all components about size of output
    for (auto &component : m_components)
    {
        component->preLaunchFrame();
    }
    m_rayGenerator->launchFrame(0, output_buffer);
    for (auto &component : m_components)
    {
        component->postLaunchFrame();
    }
}

} // end namespace opg
