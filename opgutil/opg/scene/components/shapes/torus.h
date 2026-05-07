#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "opg/scene/components/shapes/torus.cuh"

namespace opg {

class TorusShape : public Shape
{
public:
    OPG_API TorusShape(PrivatePtr<Scene> scene, const Properties &_props);
    virtual ~TorusShape();

    virtual void prepareAccelerationStructure() override;

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override;

protected:
    TorusShapeData               m_data;
    DeviceBuffer<TorusShapeData> m_data_buffer;
};

} // end namespace opg
