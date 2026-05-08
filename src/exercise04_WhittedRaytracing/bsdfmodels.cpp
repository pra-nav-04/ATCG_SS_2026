#include "bsdfmodels.h"

#include "opg/opg.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/scene/sceneloader.h"


OpaqueBSDF::OpaqueBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    BSDF(std::move(_scene), _props)
{
    m_bsdf_data.diffuse_color = glm::vec3(_props.getVector("diffuse_color", glm::vec4(0)));
    m_bsdf_data.specular_F0 = glm::vec3(_props.getVector("specular_f0", glm::vec4(0)));
    m_bsdf_data_buffer.alloc(1);
    m_bsdf_data_buffer.upload(&m_bsdf_data);
}

OpaqueBSDF::~OpaqueBSDF()
{
}

void OpaqueBSDF::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "bsdfmodels.cu");
    OptixProgramGroup eval_bsdf_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__opaque_evalBSDF" });
    OptixProgramGroup sample_bsdf_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__opaque_sampleBSDF" });

    uint32_t eval_bsdf_index = sbt->addCallableEntry(eval_bsdf_prog_group, m_bsdf_data_buffer.data());
    uint32_t sample_bsdf_index = sbt->addCallableEntry(sample_bsdf_prog_group, m_bsdf_data_buffer.data());

    BSDFVPtrTable vptr_table_data;
    vptr_table_data.component_flags = BSDFComponentFlag::DiffuseReflection | BSDFComponentFlag::IdealReflection;
    vptr_table_data.evalCallIndex = eval_bsdf_index;
    vptr_table_data.sampleCallIndex = sample_bsdf_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}


RefractiveBSDF::RefractiveBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props) :
    BSDF(std::move(_scene), _props)
{
    // index_of_refraction = ior_ext / ior_int
    // where ior_ext is the refractive index of the external medium and ior_int is the
    // refractive index of the internal medium as defined by the surface normal pointing outwards.
    m_bsdf_data.index_of_refraction = _props.getFloat("ior", 1.5f);
    m_bsdf_data_buffer.alloc(1);
    m_bsdf_data_buffer.upload(&m_bsdf_data);
}

RefractiveBSDF::~RefractiveBSDF()
{
}

void RefractiveBSDF::initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt)
{
    auto ptx_filename = opg::getPtxFilename(OPG_TARGET_NAME, "bsdfmodels.cu");
    OptixProgramGroup eval_bsdf_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__refractive_evalBSDF" });
    OptixProgramGroup sample_bsdf_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__refractive_sampleBSDF" });

    uint32_t eval_bsdf_index = sbt->addCallableEntry(eval_bsdf_prog_group, m_bsdf_data_buffer.data());
    uint32_t sample_bsdf_index = sbt->addCallableEntry(sample_bsdf_prog_group, m_bsdf_data_buffer.data());

    BSDFVPtrTable vptr_table_data;
    vptr_table_data.component_flags = BSDFComponentFlag::IdealReflection | BSDFComponentFlag::IdealTransmission;
    vptr_table_data.evalCallIndex = eval_bsdf_index;
    vptr_table_data.sampleCallIndex = sample_bsdf_index;
    m_vptr_table.allocIfRequired(1);
    m_vptr_table.upload(&vptr_table_data);
}


namespace opg {

OPG_REGISTER_SCENE_COMPONENT_FACTORY(OpaqueBSDF, "bsdf.opaque");
OPG_REGISTER_SCENE_COMPONENT_FACTORY(RefractiveBSDF, "bsdf.refractive");

} // end namespace opg
