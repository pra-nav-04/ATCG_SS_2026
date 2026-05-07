#pragma once

#include "opg/scene/interface/raygenerator.h"
#include "radiosityemitter.h"

#include "radiosityraygenerator.cuh"

class RadiosityRayGenerator : public opg::RayGenerator
{
public:
    RadiosityRayGenerator(PrivatePtr<opg::Scene> scene, const opg::Properties &props);
    ~RadiosityRayGenerator();

    virtual void launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer) override;

    virtual void finalize() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    void allocateRadiosityBuffers();

    void computeFormFactor();
    void computeRadiosity();

private:
    std::vector<RadiosityEmitter *>             m_radiosityEmitters;
    opg::DeviceBuffer<RadiosityLaunchParams>    m_launch_params;


    size_t                       m_total_primitive_count;
    opg::DeviceBuffer<float>     m_form_factor_matrix_buffer;


    uint32_t m_generateRadiosityRaygenIndex   = 0;
    uint32_t m_renderRadiosityRaygenIndex     = 0;
    uint32_t m_surfaceMissIndex               = 0;
    uint32_t m_occlusionMissIndex             = 0;
};
