#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "opg/scene/components/shapes/cylinder.cuh"

namespace opg {

class CylinderShape : public Shape
{
public:
    OPG_API CylinderShape(PrivatePtr<Scene> scene, const Properties &_props);
    virtual ~CylinderShape();

    virtual void prepareAccelerationStructure() override;

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override;
};

} // end namespace opg
