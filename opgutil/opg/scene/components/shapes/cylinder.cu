#include "opg/scene/components/shapes/cylinder.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"

#include "opg/raytracing/optixglm.h"


extern "C" __global__ void __intersection__cylinder()
{
    const glm::vec3 center = glm::vec3(0, 0, 0);
    const float     radius = 1;
    const float     half_height = 1.0f;
    const glm::vec3 axis   = glm::vec3(0, 0, 1);

    const glm::vec3 ray_orig = optixGetObjectRayOriginGLM();
    const glm::vec3 ray_dir  = optixGetObjectRayDirectionGLM();
    const float     ray_tmin = optixGetRayTmin();
    const float     ray_tmax = optixGetRayTmax();

    glm::vec3 O = ray_orig - center;
    glm::vec3 D = ray_dir;

    // project ray_dir and ray_origin into plane orthogonal to cylinder axis
    glm::vec3 O_proj = O - glm::dot(O, axis) / glm::dot(axis, axis) * axis;
    glm::vec3 D_proj = D - glm::dot(D, axis) / glm::dot(axis, axis) * axis;

    float p = glm::dot(D_proj, O_proj) / glm::dot(D_proj, D_proj); // p/2 actually
    float q = (glm::dot(O_proj, O_proj) - radius * radius) / glm::dot(D_proj, D_proj);

    float k = p*p - q;
    if (k < 0)
        return;

    // Try to report first interesction
    float t = -p - glm::sqrt(k);
    if (t > ray_tmin && t < ray_tmax)
    {
        // Check if the intersection point is within the height limit of the cylinder as specified by the axis length
        float z = glm::dot(O + t * D, axis) / glm::dot(axis, axis);
        if (glm::abs(z) <= half_height)
            if (optixReportIntersection(t, 0))
                return;
    }

    // Report second intersection
    t = -p + glm::sqrt(k);
    if (t > ray_tmin && t < ray_tmax)
    {
        // Check if the intersection point is within the height limit of the cylinder as specified by the axis length
        float z = glm::dot(O + t * D, axis) / glm::dot(axis, axis);
        if (glm::abs(z) <= half_height)
            optixReportIntersection(t, 0);
    }
}

extern "C" __global__ void __closesthit__cylinder()
{
    SurfaceInteraction *si = getPayloadDataPointer<SurfaceInteraction>();
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();

    // NOTE: optixGetObjectRayOrigin() and optixGetObjectRayDirection() are not available in closest hit programs.
    // const glm::vec3 object_ray_origin = optixGetObjectRayOriginGLM();
    // const glm::vec3 object_ray_dir    = optixGetObjectRayDirectionGLM();

    const glm::vec3 local_axis = glm::vec3(0, 0, 1);
    const float half_height = 1.0f;


    // Set incoming ray direction and distance
    si->incoming_ray_dir = world_ray_dir;
    si->incoming_distance = tmax;

    // World space postion
    si->position = world_ray_origin + tmax * world_ray_dir;

    // Transform position into local object space to compute normals and friends
    glm::vec3 local_position = optixTransformPointFromWorldToObjectSpace(si->position);

    // Object space normal is proportional to vector from center to position
    const glm::vec3 local_normal = glm::normalize(local_position - glm::dot(local_position, local_axis) / glm::dot(local_axis, local_axis) * local_axis);
    // Tangent corresponds to vector orthogonal to "up" vector local center
    const glm::vec3 local_tangent = glm::normalize(glm::cross(local_axis, local_normal));

    // Transform local object space normal to world space normal
    si->normal = local_normal;
    si->normal = optixTransformNormalFromObjectToWorldSpace(si->normal);
    si->normal = glm::normalize(si->normal);
    si->geom_normal = si->normal;

    // Transform local opbject space tangent to world space tangent
    si->tangent = local_tangent;
    si->tangent = optixTransformNormalFromObjectToWorldSpace(si->tangent);
    si->tangent = glm::normalize(si->tangent);

    float z = glm::dot(local_position, local_axis) / half_height;
    float phi = glm::atan2(local_tangent.y, local_tangent.x);

    // UV coordinates correspond to (normalized) cylindrical coordinates
    si->uv = glm::vec2(phi / glm::two_pi<float>(), z * 0.5f + 0.5f);

    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
    si->inside_medium = sbt_data->inside_medium;
    si->outside_medium = sbt_data->outside_medium;
}


extern "C" __device__ ShapeInstanceSamplingResult __direct_callable__cylinder_sample_position()
{
    /// ShapeInstanceSamplingResult result = {};

    // TODO

    // return result;
}

extern "C" __device__ float __direct_callable__cylinder_eval_position_sampling_pdf(const glm::vec3 &sampled_position)
{
    // TODO
    // return 0;// 1/area;
}

extern "C" __device__ ShapeInstanceSamplingResult __direct_callable__cylinder_sample_next_event(const SurfaceInteraction &si)
{
    return __direct_callable__cylinder_sample_position();
}

extern "C" __device__ float __direct_callable__cylinder_eval_next_event_sampling_pdf(const SurfaceInteraction &si, const glm::vec3 &sampled_position)
{
    return __direct_callable__cylinder_eval_position_sampling_pdf(sampled_position);
}
