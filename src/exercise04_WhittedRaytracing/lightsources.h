#pragma once

#include "opg/scene/interface/emitter.h"
#include "opg/memory/devicebuffer.h"

#include "lightsources.cuh"

class PointLight : public opg::Emitter
{
public:
    PointLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~PointLight();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt);

protected:
    PointLightData m_data;
    opg::DeviceBuffer<PointLightData> m_data_buffer;
};

class DirectionalLight : public opg::Emitter
{
public:
    DirectionalLight(PrivatePtr<opg::Scene> _scene, const opg::Properties &_props);
    virtual ~DirectionalLight();

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt);

protected:
    DirectionalLightData m_data;
    opg::DeviceBuffer<DirectionalLightData> m_data_buffer;
};
