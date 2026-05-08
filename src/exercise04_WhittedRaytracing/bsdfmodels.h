#pragma once

#include "opg/scene/interface/bsdf.h"

#include "bsdfmodels.cuh"

class OpaqueBSDF : public opg::BSDF
{
public:
    OpaqueBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~OpaqueBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    OpaqueBSDFData m_bsdf_data;
    opg::DeviceBuffer<OpaqueBSDFData> m_bsdf_data_buffer;
};

class RefractiveBSDF : public opg::BSDF
{
public:
    RefractiveBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~RefractiveBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    RefractiveBSDFData m_bsdf_data;
    opg::DeviceBuffer<RefractiveBSDFData> m_bsdf_data_buffer;
};
