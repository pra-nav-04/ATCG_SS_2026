#include "opg/scene/components/shapes/sphere.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"

#include "opg/raytracing/optixglm.h"


extern "C" __global__ void __intersection__sphere()
{
    // Sphere position is (0, 0, 0)
    // Sphere radius is 1
    const glm::vec3 sphere_center = glm::vec3(0, 0, 0);
    const float     radius = 1;

    const glm::vec3 ray_orig = optixGetObjectRayOriginGLM();
    const glm::vec3 ray_dir  = optixGetObjectRayDirectionGLM();
    const float     ray_tmin = optixGetRayTmin();
    const float     ray_tmax = optixGetRayTmax();

    glm::vec3 O = ray_orig - sphere_center;
    glm::vec3 D = ray_dir;

    float p = glm::dot(D, O) / glm::dot(D, D); // p/2 actually
    float q = (glm::dot(O, O) - radius * radius) / glm::dot(D, D);

    float k = p*p - q;
    if (k < 0)
        return;

    // Try to report first interesction
    float t = -p - glm::sqrt(k);
    if ( t > ray_tmin && t < ray_tmax )
        if (optixReportIntersection( t, 0 ))
            return;

    // Report second intersection
    t = -p + glm::sqrt(k);
    if ( t > ray_tmin && t < ray_tmax )
        optixReportIntersection( t, 0 );
}

extern "C" __global__ void __closesthit__sphere()
{
    SurfaceInteraction*                 si       = getPayloadDataPointer<SurfaceInteraction>();
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();

    // NOTE: optixGetObjectRayOrigin() and optixGetObjectRayDirection() are not available in closest hit programs.
    // const glm::vec3 object_ray_origin = optixGetObjectRayOriginGLM();
    // const glm::vec3 object_ray_dir    = optixGetObjectRayDirectionGLM();


    // Set incoming ray direction and distance
    si->incoming_ray_dir = world_ray_dir;
    si->incoming_distance = tmax;

    // World space postion
    si->position = world_ray_origin + tmax * world_ray_dir;

    // Transform position into local object space to compute normals and friends
    glm::vec3 local_position = optixTransformPointFromWorldToObjectSpace(si->position);

    // const glm::vec3 local_position = object_ray_origin + tmax * object_ray_dir;
    const glm::vec3 local_up = glm::vec3(0, 0, 1);
    // Object space normal is identical to position since the origin is at (0,0,0)
    const glm::vec3 local_normal = glm::normalize(local_position);
    // Tangent corresponds to longitutde vector, orthogonal to "up" vector and normal
    const glm::vec3 local_tangent = glm::normalize(glm::cross(local_up, local_normal));

    // Transform local object space normal to world space normal
    si->normal = local_normal;
    si->normal = optixTransformNormalFromObjectToWorldSpace(si->normal);
    si->normal = glm::normalize(si->normal);
    si->geom_normal = si->normal;

    // Transform local opbject space tangent to world space tangent
    si->tangent = local_tangent;
    si->tangent = optixTransformNormalFromObjectToWorldSpace(si->tangent);
    si->tangent = glm::normalize(si->tangent);

    float theta = glm::acos(local_normal.z);
    float phi = glm::atan2(local_normal.y, local_normal.x);

    // UV coordinates correspond to (normalized) longitude and lattitude coordinates
    si->uv = glm::vec2(phi / glm::two_pi<float>(), theta / glm::pi<float>());

    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
    si->inside_medium = sbt_data->inside_medium;
    si->outside_medium = sbt_data->outside_medium;
}




extern "C" __device__ ShapeInstanceSamplingResult __direct_callable__sphere_sample_position()
{
    /// ShapeInstanceSamplingResult result = {};

    // TODO

    // return result;
}

extern "C" __device__ float __direct_callable__sphere_eval_position_sampling_pdf(const glm::vec3 &sampled_position)
{
    // TODO
    // return 0;// 1/area;
}

extern "C" __device__ ShapeInstanceSamplingResult __direct_callable__sphere_sample_next_event(const SurfaceInteraction &si)
{
    return __direct_callable__sphere_sample_position();
}

extern "C" __device__ float __direct_callable__sphere_eval_next_event_sampling_pdf(const SurfaceInteraction &si, const glm::vec3 &sampled_position)
{
    return __direct_callable__sphere_eval_position_sampling_pdf(sampled_position);
}
