#include "lightsources.h"

#include "opg/opg.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"


PointLight::PointLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props)
{
    float intensity = _props.getFloat("intensity", 1);
    glm::vec3 color = glm::vec3(_props.getVector("color", glm::vec4(1)));
    m_data.intensity = intensity * color;
    m_data.position = glm::vec3(_props.getVector("position", glm::vec4(0)));
    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);
}

PointLight::~PointLight()
{
}

void PointLight::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "lightsources.cu");
    OptixProgramGroup sample_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__pointlight_sampleLight" });

    uint32_t sample_light_index = sbt->addCallableEntry(sample_light_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = +EmitterFlag::InfinitesimalSize;
    vptr_table_data.sampleCallIndex = sample_light_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}


DirectionalLight::DirectionalLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    Emitter(std::move(_scene), _props)
{
    float irradiance = _props.getFloat("irradiance", 1);
    glm::vec3 color = glm::vec3(_props.getVector("color", glm::vec4(1)));
    m_data.irradiance_at_receiver = irradiance * color;
    m_data.direction = glm::normalize(glm::vec3(_props.getVector("direction", glm::vec4(0, 0, 1, 0))));
    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);
}

DirectionalLight::~DirectionalLight()
{
}

void DirectionalLight::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "lightsources.cu");
    OptixProgramGroup sample_light_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__directionallight_sampleLight" });

    uint32_t sample_light_index = sbt->addCallableEntry(sample_light_prog_group, m_data_buffer.data());

    EmitterVPtrTable vptr_table_data;
    vptr_table_data.flags = EmitterFlag::InfinitesimalSize | EmitterFlag::DistantEmitter;
    vptr_table_data.sampleCallIndex = sample_light_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}

namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(PointLight, "emitter.point");
OPG_REGISTER_SCENE_COMPONENT_FACTORY(DirectionalLight, "emitter.directional");

} // end namespace opg
