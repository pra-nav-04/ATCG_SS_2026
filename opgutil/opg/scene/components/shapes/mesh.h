#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/shape.h"
#include "opg/scene/components/shapeinstance.h"

#include "opg/scene/components/shapes/mesh.cuh"

namespace opg {

class MeshShape : public Shape
{
public:
    OPG_API MeshShape(PrivatePtr<Scene> scene, const Properties &_props);
    virtual ~MeshShape();

    virtual void prepareAccelerationStructure() override;

    const MeshShapeData &getMeshShapeData() const { return m_data; }

protected:
    struct InitInDerivedClass {};
    OPG_API MeshShape(PrivatePtr<Scene> scene, const Properties &_props, InitInDerivedClass);

    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override;

protected:
    MeshShapeData                m_data;
    DeviceBuffer<MeshShapeData>  m_data_buffer;
};

OPG_API SceneComponent *createMeshFromObjFile(Scene *scene, const Properties &props);

} // end namespace opg
