#pragma once

#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/interface/emitter.cuh"
#include "opg/memory/devicebuffer.h"

namespace opg {

class ShapeInstance;

// An Emitter component is an interface defines how light is emitted in the scene.
// An emitter can either be standalone or be attached to a shape instance if it represents an emissive surface.
// Components deriving from this class must populate the fields in `m_vptr_table` to
// define "function pointers" that are called on the GPU when this Emitter should be evaluated.
class Emitter : public SceneComponent
{
public:
    Emitter(PrivatePtr<Scene> _scene, const Properties &_props) : SceneComponent(std::move(_scene), _props) {}
    virtual ~Emitter() {}

    virtual void assignShapeInstance(PrivatePtr<ShapeInstance> instance) { throw std::runtime_error("This emitter cannot be assigned to a shape instance!"); };

    const EmitterVPtrTable *getEmitterVPtrTable() const { return m_vptr_table.data(); }

    virtual float getTotalEmittedPower() const { return 1.0f; } // < Dummy implementation that "weights" this light source with 1.

    virtual bool isEnvironment() const { return false; }

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override = 0; // this must be overriden by children

protected:
    DeviceBuffer<EmitterVPtrTable> m_vptr_table;
};

} // end namespace opg
