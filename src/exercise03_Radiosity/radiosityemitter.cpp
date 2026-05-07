#include "radiosityemitter.h"

#include "opg/opg.h"
#include "opg/scene/properties.h"

#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"

RadiosityEmitter::RadiosityEmitter(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props),
    m_shapeInstance(nullptr),
    m_mesh(nullptr),
    m_albedo(_props.getVector("albedo", glm::vec4(1))),
    m_emission(_props.getVector("emission", glm::vec4(0)))
{
}

RadiosityEmitter::~RadiosityEmitter()
{
}

void RadiosityEmitter::assignShapeInstance(PrivatePtr<opg::ShapeInstance> instance)
{
    OPG_CHECK_MSG(m_shapeInstance == nullptr, "Radiosity emitter can only be attached to a single shape instance!");

    m_shapeInstance = instance;
    m_mesh = dynamic_cast<opg::MeshShape*>(m_shapeInstance->getShape());

    OPG_CHECK_MSG(m_mesh != nullptr, "Radiosity emitter only works on meshes");

    m_primitiveCount = m_mesh->getMeshShapeData().indices.count;
    m_primitiveRadiosity.alloc(m_primitiveCount);

    std::vector<glm::vec3> radiosityData(m_primitiveCount);
    for (int i = 0; i < radiosityData.size(); ++i)
    {
        radiosityData[i] = (i % 2 == 0) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    }
    m_primitiveRadiosity.upload(radiosityData.data());

    m_data.albedo = m_albedo;
    m_data.emission = m_emission;
    m_data.primitiveRadiosity = m_primitiveRadiosity.view();

    m_data_buffer.allocIfRequired(1);
    m_data_buffer.upload(&m_data);
}

void RadiosityEmitter::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    // TODO method for sampling a point in a triangle (primitive)?

    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "radiosityemitter.cu");
    OptixProgramGroup eval_radiosity_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__evalRadiosity" });

    uint32_t eval_radiosity_index = sbt->addCallableEntry(eval_radiosity_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = EmitterFlag::InfinitesimalSize | EmitterFlag::DistantEmitter;
    vptr_table_data.evalCallIndex = eval_radiosity_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}

namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(RadiosityEmitter, "emitter.radiosity");

}
