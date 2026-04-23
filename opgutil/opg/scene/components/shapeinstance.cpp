#include "opg/scene/components/shapeinstance.h"

#include "opg/scene/sceneloader.h"

#include "opg/raytracing/shaderbindingtable.h"

namespace opg {

ShapeInstance::ShapeInstance(PrivatePtr<Scene> _scene, const Properties &_props) :
    SceneComponent(std::move(_scene), _props),
    m_transform{ _props.getMatrix("to_world", glm::mat4(1)) },
    m_shape{ _props.getComponentAs<Shape>("shape") },
    m_bsdf{ _props.getComponentAs<BSDF>("bsdf") },
    m_emitter{ _props.getComponentAs<Emitter>("emitter") },
    m_inside_medium{ _props.getComponentAs<Medium>("inside_medium") },
    m_outside_medium{ _props.getComponentAs<Medium>("outside_medium") }
{
    if (m_emitter != nullptr)
        m_emitter->assignShapeInstance(PrivatePtr<ShapeInstance>(this));
}

ShapeInstance::~ShapeInstance()
{
}

void ShapeInstance::initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt)
{
    m_shape->ensurePipelineInitialized(pipeline, sbt);
    if (m_bsdf != nullptr)
        m_bsdf->ensurePipelineInitialized(pipeline, sbt);
    if (m_emitter != nullptr)
        m_emitter->ensurePipelineInitialized(pipeline, sbt);
    if (m_inside_medium != nullptr)
        m_inside_medium->ensurePipelineInitialized(pipeline, sbt);
    if (m_outside_medium != nullptr)
        m_outside_medium->ensurePipelineInitialized(pipeline, sbt);

    ShapeInstanceHitGroupSBTData sbtData;
    sbtData.shape = m_shape->m_data_ptr;
    sbtData.bsdf = m_bsdf != nullptr ? m_bsdf->getBSDFVPtrTable() : nullptr;
    sbtData.emitter = m_emitter != nullptr ? m_emitter->getEmitterVPtrTable() : nullptr;
    sbtData.inside_medium = m_inside_medium != nullptr ? m_inside_medium->getMediumVPtrTable() : nullptr;
    sbtData.outside_medium = m_outside_medium != nullptr ? m_outside_medium->getMediumVPtrTable() : nullptr;
    sbt->addHitEntry(m_shape->m_hit_prog_group, sbtData);

    // TODO add sampling callable functions!
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY(ShapeInstance, "shapeinstance");

} // end namespace opg
