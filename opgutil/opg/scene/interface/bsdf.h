#pragma once

#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/interface/bsdf.cuh"
#include "opg/memory/devicebuffer.h"

namespace opg {

// A BSDF component is an interface defines how light is reflected off a surface.
// Components deriving from this class must populate the fields in `m_vptr_table` to
// define "function pointers" that are called on the GPU when this BSDF should be evaluated.
class BSDF : public SceneComponent
{
public:
    BSDF(PrivatePtr<Scene> _scene, const Properties &_props) : SceneComponent(std::move(_scene), _props), m_vptr_table(1) {}
    virtual ~BSDF() {}

    const BSDFVPtrTable *getBSDFVPtrTable() const { return m_vptr_table.data(); }

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override = 0; // this must be overriden by children

protected:
    DeviceBuffer<BSDFVPtrTable> m_vptr_table;
};

} // end namespace opg
