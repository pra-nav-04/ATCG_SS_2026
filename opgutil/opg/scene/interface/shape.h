#pragma once

#include "opg/scene/interface/scenecomponent.h"
#include "opg/memory/devicebuffer.h"
#include "opg/glmwrapper.h"

#include "opg/scene/components/shapeinstance.cuh"

namespace opg {

// Scene components that define geometry, e.g. meshes or analytical spheres, derive from the Shape class.
class Shape : public SceneComponent
{
public:
    Shape(PrivatePtr<Scene> _scene, const Properties &_props) : SceneComponent(std::move(_scene), _props) {}
    virtual ~Shape() {}

    inline OptixTraversableHandle getAST() { return m_ast_handle; }

public:
    virtual void prepareAccelerationStructure() = 0; // this must be overridden by children

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override = 0; // this must be overriden by children

protected:
    friend class ShapeInstance;

    // Ray intersection acceleration structure for this shape:
    GenericDeviceBuffer     m_ast_buffer;
    OptixTraversableHandle  m_ast_handle        = 0;

    // Pointer to the shape data in device memory
    ShapeData*              m_data_ptr = nullptr;

    // Ray tracing hit program containin the intersection, any hit and closest hit shader code:
    OptixProgramGroup       m_hit_prog_group    = 0;
    // Callable programs for sampling positions on surface of shape (if available):
    OptixProgramGroup       m_sample_position_call_prog_group               = 0;
    OptixProgramGroup       m_eval_position_sampling_pdf_call_prog_group    = 0;
    OptixProgramGroup       m_sample_next_event_call_prog_group             = 0;
    OptixProgramGroup       m_eval_next_event_sampling_pdf_call_prog_group  = 0;
};

} // end namespace opg
