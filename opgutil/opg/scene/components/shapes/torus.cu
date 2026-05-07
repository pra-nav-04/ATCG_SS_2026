#include "opg/scene/components/shapes/torus.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"

#include "opg/raytracing/optixglm.h"
#include "opg/hostdevice/roots3and4.h"


extern "C" __global__ void __intersection__torus()
{
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());
    const TorusShapeData* torus_data = reinterpret_cast<const TorusShapeData *>(sbt_data->shape);

    const glm::vec3 torus_center = glm::vec3(0, 0, 0);
    const float     major_radius = 1;
    const float     minor_radius = torus_data->minorRadius;

    const glm::vec3 ray_orig = optixGetObjectRayOriginGLM();
    const glm::vec3 ray_dir  = optixGetObjectRayDirectionGLM();
    const float     ray_tmin = optixGetRayTmin();
    const float     ray_tmax = optixGetRayTmax();

    glm::dvec3 O = ray_orig - torus_center;
    glm::dvec3 D = ray_dir;

    double DD = glm::dot(D, D);
    double DO = glm::dot(D, O);
    double OO = glm::dot(O, O);

    double OO_rrRR = OO - (minor_radius*minor_radius + major_radius*major_radius);

    double c4 = DD * DD;
    double c3 = 4 * DD * DO;
    double c2 = 2 * DD * OO_rrRR + 4 * DO*DO + 4 * major_radius*major_radius * D.z*D.z;
    double c1 = 4 * OO_rrRR * DO + 8* major_radius * major_radius*O.z*D.z;
    double c0 = OO_rrRR * OO_rrRR - 4*major_radius*major_radius * (minor_radius*minor_radius - O.z*O.z);

    double c[5] = {c0, c1, c2, c3, c4};
    double s[4];
    int rootCount = SolveQuartic<double>(c, s);
    // NOTE: roots are not sorted!

    #define COMP_SWAP(a, b) do { if (s[a] > s[b]) { auto c = s[a]; s[a] = s[b]; s[b] = c; } } while (0)

    if (rootCount >= 2)
    {
        COMP_SWAP(0, 1);
        if (rootCount >= 4)
        {
            COMP_SWAP(2, 3);
            COMP_SWAP(0, 3);
        }
        if (rootCount >= 3)
        {
            COMP_SWAP(1, 2);
            COMP_SWAP(0, 1);
        }
        if (rootCount >= 4)
        {
            COMP_SWAP(2, 3);
        }
    }


    // Try to report first interesction
    if ( rootCount >= 1 && s[0] > ray_tmin && s[0] < ray_tmax )
        if (optixReportIntersection( s[0], 0 ))
            return;

    if ( rootCount >= 2 && s[1] > ray_tmin && s[1] < ray_tmax )
        if (optixReportIntersection( s[1], 0 ))
            return;

    if ( rootCount >= 3 && s[2] > ray_tmin && s[2] < ray_tmax )
        if (optixReportIntersection( s[2], 0 ))
            return;

    if ( rootCount >= 4 && s[3] > ray_tmin && s[3] < ray_tmax )
        if (optixReportIntersection( s[3], 0 ))
            return;
}

extern "C" __global__ void __closesthit__torus()
{
    SurfaceInteraction *si = getPayloadDataPointer<SurfaceInteraction>();
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

    const glm::vec3 local_up = glm::vec3(0, 0, 1);

    // Center of the torus tube closest to position
    glm::vec3 local_center = glm::normalize(glm::vec3(local_position.x, local_position.y, 0));

    // Object space normal is proportional to vector from center to position
    const glm::vec3 local_normal = glm::normalize(local_position - local_center);
    // Tangent corresponds to vector orthogonal to "up" vector local center
    const glm::vec3 local_tangent = glm::normalize(glm::cross(local_up, local_center));

    // Transform local object space normal to world space normal
    si->normal = local_normal;
    si->normal = optixTransformNormalFromObjectToWorldSpace(si->normal);
    si->normal = glm::normalize(si->normal);
    si->geom_normal = si->normal;

    // Transform local opbject space tangent to world space tangent
    si->tangent = local_tangent;
    si->tangent = optixTransformNormalFromObjectToWorldSpace(si->tangent);
    si->tangent = glm::normalize(si->tangent);

    float theta = glm::atan2(glm::dot(local_position, local_up), -glm::dot(local_position - local_center, glm::normalize(local_center)));
    float phi = glm::atan2(local_tangent.y, local_tangent.x);

    // UV coordinates correspond to (normalized) longitude and lattitude coordinates
    si->uv = glm::vec2(phi / glm::two_pi<float>(), theta / glm::two_pi<float>());

    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
    si->inside_medium = sbt_data->inside_medium;
    si->outside_medium = sbt_data->outside_medium;
}


extern "C" __device__ ShapeInstanceSamplingResult __direct_callable__torus_sample_position()
{
    /// ShapeInstanceSamplingResult result = {};

    // TODO

    // return result;
}

extern "C" __device__ float __direct_callable__torus_eval_position_sampling_pdf(const glm::vec3 &sampled_position)
{
    // TODO
    // return 0;// 1/area;
}

extern "C" __device__ ShapeInstanceSamplingResult __direct_callable__torus_sample_next_event(const SurfaceInteraction &si)
{
    return __direct_callable__torus_sample_position();
}

extern "C" __device__ float __direct_callable__torus_eval_next_event_sampling_pdf(const SurfaceInteraction &si, const glm::vec3 &sampled_position)
{
    return __direct_callable__torus_eval_position_sampling_pdf(sampled_position);
}
