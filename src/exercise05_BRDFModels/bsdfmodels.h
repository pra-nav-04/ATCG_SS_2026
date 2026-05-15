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

class PhongBSDF : public opg::BSDF
{
public:
    PhongBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~PhongBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    PhongBSDFData m_bsdf_data;
    opg::DeviceBuffer<PhongBSDFData> m_bsdf_data_buffer;
};

class WardBSDF : public opg::BSDF
{
public:
    WardBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~WardBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    WardBSDFData m_bsdf_data;
    opg::DeviceBuffer<WardBSDFData> m_bsdf_data_buffer;
};

class GGXBSDF : public opg::BSDF
{
public:
    GGXBSDF(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~GGXBSDF();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

protected:
    GGXBSDFData m_bsdf_data;
    opg::DeviceBuffer<GGXBSDFData> m_bsdf_data_buffer;
};
