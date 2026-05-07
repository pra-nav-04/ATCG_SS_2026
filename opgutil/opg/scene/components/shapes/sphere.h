#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "opg/scene/components/shapes/sphere.cuh"

namespace opg {

class SphereShape : public Shape
{
public:
    OPG_API SphereShape(PrivatePtr<Scene> scene, const Properties &_props);
    virtual ~SphereShape();

    virtual void prepareAccelerationStructure() override;

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override;
};

} // end namespace opg
