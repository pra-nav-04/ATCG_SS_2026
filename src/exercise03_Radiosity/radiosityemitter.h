#pragma once

#include "opg/scene/interface/emitter.h"
#include "opg/scene/components/shapeinstance.h"
#include "opg/scene/components/shapes/mesh.h"

#include "radiosityemitter.cuh"

class RadiosityEmitter : public opg::Emitter
{
public:
    RadiosityEmitter(PrivatePtr<opg::Scene> scene, const opg::Properties &props);
    virtual ~RadiosityEmitter();

    virtual void assignShapeInstance(PrivatePtr<opg::ShapeInstance> instance) override;

    inline opg::ShapeInstance*  getShapeInstance() const { return m_shapeInstance; }
    inline opg::MeshShape*      getMeshShape() const { return m_mesh; }

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    // RadiosityRayGenerator is allowed to access private data!
    friend class RadiosityRayGenerator;

    opg::ShapeInstance* m_shapeInstance;
    opg::MeshShape*     m_mesh;

    uint32_t            m_primitiveCount;
    opg::DeviceBuffer<glm::vec3> m_primitiveRadiosity;

    glm::vec3           m_albedo;
    glm::vec3           m_emission;

    RadiosityEmitterData m_data;
    opg::DeviceBuffer<RadiosityEmitterData> m_data_buffer;
};
