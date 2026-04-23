#pragma once

#include "opg/scene/interface/bsdf.cuh"
#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/interface/medium.cuh"

/*
struct ShapeData
{
    // Nothing.
    // Contents are specific to shape type!
};
*/
typedef void ShapeData;


struct ShapeInstanceTransform
{
    glm::mat4 local_to_world;
    glm::mat4 world_to_local;
};

struct ShapeInstanceHitGroupSBTData
{
    const ShapeData*        shape;
    const BSDFVPtrTable*    bsdf;
    const EmitterVPtrTable* emitter;
    const MediumVPtrTable*  inside_medium;
    const MediumVPtrTable*  outside_medium;
    // (Instance) transform is accessible through OptiX api in hit-related shaders
};

struct ShapeInstanceCallableSBTData
{
    const ShapeData*                shape;
    const ShapeInstanceTransform*   transform;
    // No BSDF or Emitter required here...
};


struct ShapeInstanceVPtrTable
{
    uint32_t samplePositionCallIndex;
    uint32_t evalPositionSamplingPdfCallIndex;

    uint32_t sampleNextEventCallIndex;
    uint32_t evalNextEventSamplingPdfCallIndex;
};

struct ShapeInstanceSamplingResult
{
    glm::vec3 position;
    glm::vec3 normal;
    //glm::vec3 tangent;
    //glm::vec2 uv;
    float     sampling_pdf;
};

//
// The following code is only valid in device code
//
#ifdef __CUDACC__
#include <optix_device.h>

struct SurfaceInteraction;

__device__ ShapeInstanceSamplingResult shapeInstanceSamplePosition(const ShapeInstanceVPtrTable &shape_instance)
{
    return optixDirectCall<ShapeInstanceSamplingResult>(shape_instance.samplePositionCallIndex);
}

__device__ float shapeInstanceEvalPositionSamplingPdf(const ShapeInstanceVPtrTable &shape_instance, const glm::vec3 &sampled_position)
{
    return optixDirectCall<float, const glm::vec3 &>(shape_instance.evalPositionSamplingPdfCallIndex, sampled_position);
}

__device__ ShapeInstanceSamplingResult shapeInstanceSampleNextEvent(const ShapeInstanceVPtrTable &shape_instance, const SurfaceInteraction &si)
{
    return optixDirectCall<ShapeInstanceSamplingResult, const SurfaceInteraction &>(shape_instance.sampleNextEventCallIndex, si);
}

__device__ float shapeInstanceEvalNextEventSamplingPdf(const ShapeInstanceVPtrTable &shape_instance, const SurfaceInteraction &si, const glm::vec3 &sampled_position)
{
    return optixDirectCall<float, const SurfaceInteraction &, const glm::vec3 &>(shape_instance.evalNextEventSamplingPdfCallIndex, si, sampled_position);
}

#endif // __CUDACC__
