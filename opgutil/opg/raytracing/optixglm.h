#pragma once

#ifdef __CUDACC__

#include "opg/glmwrapper.h"
#include <optix.h>

// Overload OptiX built-in functions with versions that take GLM vectors...

/// Transforms the point using world-to-object transformation matrix resulting from the current active transformation
/// list.
///
/// The cost of this function may be proportional to the size of the transformation list.
static __forceinline__ __device__ glm::vec3 optixTransformPointFromWorldToObjectSpace(glm::vec3 point)
{
    float3 cudapt = glm2cuda(point);
    float3 cudaresult = optixTransformPointFromWorldToObjectSpace(cudapt);
    glm::vec3 result = cuda2glm(cudaresult);
    return result;
}

/// Transforms the vector using world-to-object transformation matrix resulting from the current active transformation
/// list.
///
/// The cost of this function may be proportional to the size of the transformation list.
static __forceinline__ __device__ glm::vec3 optixTransformVectorFromWorldToObjectSpace(glm::vec3 vec)
{
    return cuda2glm(optixTransformVectorFromWorldToObjectSpace(glm2cuda(vec)));
}

/// Transforms the normal using world-to-object transformation matrix resulting from the current active transformation
/// list.
///
/// The cost of this function may be proportional to the size of the transformation list.
static __forceinline__ __device__ glm::vec3 optixTransformNormalFromWorldToObjectSpace(glm::vec3 normal)
{
    return cuda2glm(optixTransformNormalFromWorldToObjectSpace(glm2cuda(normal)));
}

/// Transforms the point using object-to-world transformation matrix resulting from the current active transformation
/// list.
///
/// The cost of this function may be proportional to the size of the transformation list.
static __forceinline__ __device__ glm::vec3 optixTransformPointFromObjectToWorldSpace(glm::vec3 point)
{
    return cuda2glm(optixTransformPointFromObjectToWorldSpace(glm2cuda(point)));
}

/// Transforms the vector using object-to-world transformation matrix resulting from the current active transformation
/// list.
///
/// The cost of this function may be proportional to the size of the transformation list.
static __forceinline__ __device__ glm::vec3 optixTransformVectorFromObjectToWorldSpace(glm::vec3 vec)
{
    return cuda2glm(optixTransformVectorFromObjectToWorldSpace(glm2cuda(vec)));
}

/// Transforms the normal using object-to-world transformation matrix resulting from the current active transformation
/// list.
///
/// The cost of this function may be proportional to the size of the transformation list.
static __forceinline__ __device__ glm::vec3 optixTransformNormalFromObjectToWorldSpace(glm::vec3 normal)
{
    return cuda2glm(optixTransformNormalFromObjectToWorldSpace(glm2cuda(normal)));
}


/// Returns the rayOrigin passed into optixTrace.
///
/// May be more expensive to call in IS and AH than their object space counterparts,
/// so effort should be made to use the object space ray in those programs.
/// Only available in IS, AH, CH, MS
static __forceinline__ __device__ glm::vec3 optixGetWorldRayOriginGLM()
{
    return cuda2glm(optixGetWorldRayOrigin());
}

/// Returns the rayDirection passed into optixTrace.
///
/// May be more expensive to call in IS and AH than their object space counterparts,
/// so effort should be made to use the object space ray in those programs.
/// Only available in IS, AH, CH, MS
static __forceinline__ __device__ glm::vec3 optixGetWorldRayDirectionGLM()
{
    return cuda2glm(optixGetWorldRayDirection());
}

/// Returns the current object space ray origin based on the current transform stack.
///
/// Only available in IS and AH.
static __forceinline__ __device__ glm::vec3 optixGetObjectRayOriginGLM()
{
    return cuda2glm(optixGetObjectRayOrigin());
}

/// Returns the current object space ray direction based on the current transform stack.
///
/// Only available in IS and AH.
static __forceinline__ __device__ glm::vec3 optixGetObjectRayDirectionGLM()
{
    return cuda2glm(optixGetObjectRayDirection());
}



/// Convenience function that returns the first two attributes as floats.
///
/// When using OptixBuildInputTriangleArray objects, during intersection the barycentric
/// coordinates are stored into the first two attribute registers.
static __forceinline__ __device__ glm::vec2 optixGetTriangleBarycentricsGLM()
{
    return cuda2glm(optixGetTriangleBarycentrics());
}

/// Available in any program, it returns the current launch index within the launch dimensions specified by optixLaunch on the host.
///
/// The raygen program is typically only launched once per launch index.
static __forceinline__ __device__ glm::uvec3 optixGetLaunchIndexGLM()
{
    return cuda2glm(optixGetLaunchIndex());
}

/// Available in any program, it returns the dimensions of the current launch specified by optixLaunch on the host.
static __forceinline__ __device__ glm::uvec3 optixGetLaunchDimensionsGLM()
{
    return cuda2glm(optixGetLaunchDimensions());
}

#endif // __CUDACC__
