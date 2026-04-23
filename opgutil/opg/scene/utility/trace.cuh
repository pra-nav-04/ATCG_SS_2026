#pragma once

#include <optix.h>

#include "opg/glmwrapper.h"
#include "opg/memory/packpointer.h"
#include "opg/preprocessor.h"



// "Miscellaneous" parameters passed into the optixTrace call are grouped in this struct.
// These parameters are usually supplied via the launch parameters from the C++ code.
// Usually there is a different set of parameters used for each "kind" of ray that is shot through the scene.
// In our case, we distinguish two ray types in this framework:
// - Occlusion:
//     These rays only query for visibility.
//     The rayFlags are set such that calls to the closest-hit program are disabled and the traversal stops as
//     soon as any intersection with the scene is found, which is not neccessarily the closest one.
// - Scene Intersection:
//     These rays are used to get information about the intersecting scene geometry, like position and normal.
struct TraceParameters
{
    // OptixVisibilityMask visibilityMask;
    // Combination of flags from `OptixRayFlags`.
    uint32_t            rayFlags;
    // Gobal offset to be applied to all accesses to hitgroup entries in the shader binding table (SBT).
    uint32_t            SBToffset;
    // When an instance acceleration structure (IAS) contains multiple geometry acceleration structures (GAS), this stride is used to advance to the entry in the SBT corresponding to the respective GAS.
    uint32_t            SBTstride;
    // The index of the miss program to use when the ray does not intersect the scene geometry.
    uint32_t            missSBTIndex;
};

//
// The following code is only valid in device code
//
#ifdef __CUDACC__

//
// Generic tracing with a payload data structure
//

// In the miss and closesthit shader stages this function can be used to get the payload data pointer.
// The memory at the data pointer can be used to pass user-defined arguments into the ray tracing call,
// and more importantly is used to store results that are passed back to the `traceWithDataPointer()` caller.
// Note that the type `T` must match the type used in the call to `traceWithDataPointer()`.
template <typename T>
__forceinline__ __device__ T *getPayloadDataPointer()
{
    // Get the pointer to the payload data
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

// Initiate a new ray tracing operation into the scene.
// The first argument is a handle to the acceleration structure used to traverse the scene.
// The `ray_origin` and `ray_direction` define the ray to be traced, where `tmin` and `tmax` limit
// the range along the ray where intersections are detected.
template <typename T>
__forceinline__ __device__ void traceWithDataPointer(
        OptixTraversableHandle handle,
        glm::vec3              ray_origin,
        glm::vec3              ray_direction,
        float                  tmin,
        float                  tmax,
        TraceParameters        trace_params,
        T*                     payload_ptr
        )
{
    uint32_t u0, u1;
    packPointer(payload_ptr, u0, u1);
    optixTrace(
            handle,
            glm2cuda(ray_origin),
            glm2cuda(ray_direction),
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),  // visibilityMask
            trace_params.rayFlags, // OPTIX_RAY_FLAG_NONE
            trace_params.SBToffset,
            trace_params.SBTstride,
            trace_params.missSBTIndex,
            u0,                      // payload 0
            u1 );                    // payload 1
    // optixTrace operation will have updated content of *payload_ptr
}


//
// Occlusion
//

// In the miss and closesthit shader stages this function can be used to *set* the occlusion status of this ray.
// By default, rays are set to be occluded and if the miss program is invoked, the ray is set to be not occluded.
__forceinline__ __device__ void setOcclusionPayload(bool occluded)
{
    // Set the payload that _this_ ray will yield
    optixSetPayload_0(static_cast<uint32_t>(occluded));
}

// In the miss and closesthit shader stages this function can be used to *get* the occlusion status of this ray.
__forceinline__ __device__ bool getOcclusionPayload()
{
    // Get the payload that _this_ ray will yield
    return static_cast<bool>(optixGetPayload_0());
}

// Initiate a new ray tracing operation into the scene that only determines the occlusion/visibility.
// The first argument is a handle to the acceleration structure used to traverse the scene.
// The `ray_origin` and `ray_direction` define the ray to be traced, where `tmin` and `tmax` limit
// the range along the ray where intersections are detected.
__forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        glm::vec3              ray_origin,
        glm::vec3              ray_direction,
        float                  tmin,
        float                  tmax,
        TraceParameters        trace_params
        )
{
    uint32_t occluded = 1u;
    optixTrace(
            handle,
            glm2cuda(ray_origin),
            glm2cuda(ray_direction),
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),  // visibilityMask
            trace_params.rayFlags,   // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            trace_params.SBToffset,
            trace_params.SBTstride,
            trace_params.missSBTIndex,
            occluded );              // payload 0
    return occluded != 0;
}

#endif // __CUDACC__
