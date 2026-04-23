#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/components/shapeinstance.cuh"

#include "opg/scene/interface/shape.h"
#include "opg/scene/interface/bsdf.h"
#include "opg/scene/interface/emitter.h"
#include "opg/scene/interface/medium.h"

namespace opg {

class ShapeInstance : public SceneComponent
{
public:
    OPG_API ShapeInstance(PrivatePtr<Scene> scene, const Properties &props);
    virtual ~ShapeInstance();

    inline const glm::mat4 &getTransform() const { return m_transform; }
    inline Shape *getShape() const { return m_shape; }
    inline BSDF *getBSDF() const { return m_bsdf; }
    inline Emitter *getEmitter() const { return m_emitter; }

    inline Medium *getInsideMedium() const { return m_inside_medium; }
    inline Medium *getOutsideMedium() const { return m_outside_medium; }

    inline const ShapeInstanceVPtrTable *getShapeInstanceVPtrTable() const { return m_vptr_table.data(); }

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override;

protected:
    glm::mat4   m_transform;
    Shape*      m_shape;
    BSDF*       m_bsdf;
    Emitter*    m_emitter;
    Medium*     m_inside_medium;
    Medium*     m_outside_medium;

    // TODO inline storage or device pointer?!
    DeviceBuffer<ShapeInstanceVPtrTable> m_vptr_table;
};

} // end namespace opg
