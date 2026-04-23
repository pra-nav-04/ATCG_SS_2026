#pragma once

#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/interface/phasefunction.h"
#include "opg/scene/interface/medium.cuh"
#include "opg/memory/devicebuffer.h"

namespace opg {

class Medium : public SceneComponent
{
public:
    Medium(PrivatePtr<Scene> _scene, const Properties &_props) : SceneComponent(std::move(_scene), _props), m_vptr_table(1)
    {
        m_phase_function = _props.getComponentAs<PhaseFunction>("phase_function");
    }
    virtual ~Medium() {}

    const MediumVPtrTable *getMediumVPtrTable() const { return m_vptr_table.data(); }

    const PhaseFunction *getPhaseFunction() const { return m_phase_function; }

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override = 0; // this must be overriden by children

protected:
    DeviceBuffer<MediumVPtrTable> m_vptr_table;
    PhaseFunction *m_phase_function;
};

} // end namespace opg
