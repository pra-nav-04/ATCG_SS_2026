#pragma once

#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/interface/phasefunction.cuh"
#include "opg/memory/devicebuffer.h"

namespace opg {

class PhaseFunction : public SceneComponent
{
public:
    PhaseFunction(PrivatePtr<Scene> _scene, const Properties &_props) : SceneComponent(std::move(_scene), _props), m_vptr_table(1) {}
    virtual ~PhaseFunction() {}

    const PhaseFunctionVPtrTable *getPhaseFunctionVPtrTable() const { return m_vptr_table.data(); }

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override = 0; // this must be overriden by children

protected:
    DeviceBuffer<PhaseFunctionVPtrTable> m_vptr_table;
};

} // end namespace opg
