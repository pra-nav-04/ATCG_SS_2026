#pragma once

#include "opg/scene/interface/raygenerator.h"
#include "opg/scene/interface/emitter.h"

#include "whittedraygenerator.cuh"

class WhittedRayGenerator : public opg::RayGenerator
{
public:
    WhittedRayGenerator(PrivatePtr<opg::Scene> scene, const opg::Properties &props);
    ~WhittedRayGenerator();

    virtual void launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer) override;

    virtual void finalize() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    opg::DeviceBuffer<const EmitterVPtrTable*>  m_emitters_buffer;
    opg::DeviceBuffer<WhittedLaunchParams>      m_launch_params;
};
