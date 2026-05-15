#include <cuda_runtime.h>
#include <optix.h>
#include "opg/raytracing/optixglm.h"

#include "whittedraygenerator.cuh"

#include "opg/hostdevice/ray.h"
#include "opg/hostdevice/color.h"
#include "opg/memory/stack.h"
#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"
#include "opg/scene/interface/bsdf.cuh"
#include "opg/scene/interface/emitter.cuh"

__constant__ WhittedLaunchParams params;

extern "C" __global__ void __miss__main()
{
    SurfaceInteraction *si = getPayloadDataPointer<SurfaceInteraction>();

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();

    si->incoming_ray_dir = world_ray_dir;

    // No valid interaction found, set incoming_distance to NaN
    si->set_invalid();
}

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}

// The context of a ray holds all relevant variables that need to "survive" each ray tracing iteration.
struct RayContext
{
    // The ray that we are currently shooting through the scene.
    opg::Ray    ray;
    // How much is the radiance going to be attenuated between the current ray and the camera (based on all previous interactions).
    glm::vec3   throughput;
    // The depth of this ray, i.e. the number of bounces between the camera and this ray
    uint32_t    depth;
};


extern "C" __global__ void __raygen__main()
{
    const glm::uvec3 launch_idx  = optixGetLaunchIndexGLM();
    const glm::uvec3 launch_dims = optixGetLaunchDimensionsGLM();

    // Index of current pixel
    const glm::uvec2 pixel_index = glm::uvec2(launch_idx.x, launch_idx.y);

    // Radiance accumulated for the pixel
    glm::vec3 output_radiance = glm::vec3(0);

    opg::Stack<RayContext, RAY_STACK_SIZE> stack_of_rays_to_trace;

    // Whitted raytracing is fully deterministic, so we do not need random numbers.
    // However, we use the same interface here that is used in later exercises to sample random positions or directions from BSDFs and emitters, which requires access to a random number generator.
    // Since a random number generator has to be passed through the interfaces, but it is not used here, we just create a dummy random number generator here.
    PCG32 dummy_rng;

    // Create the initial ray from the camera
    {
        glm::vec2 uv = (glm::vec2(pixel_index)+0.5f) / glm::vec2(params.image_width, params.image_height);
        uv = 2.0f*uv - 1.0f; // [0, 1] -> [-1, 1]

        RayContext camera_ray_ctx;
        spawn_camera_ray(params.camera, uv, camera_ray_ctx.ray.origin, camera_ray_ctx.ray.direction);
        camera_ray_ctx.throughput = glm::vec3(1);
        camera_ray_ctx.depth = 0;
        stack_of_rays_to_trace.push(camera_ray_ctx);
    }

    // Bound the number of trace operations that are executed in any case.
    // Accidentally creating infinite loops on the GPU is unpleasant.
    for (int i = 0; i < MAX_TRACE_OPS; ++i)
    {
        if (stack_of_rays_to_trace.empty())
            break;

        RayContext ray_ctx = stack_of_rays_to_trace.pop();

        SurfaceInteraction si;
        traceWithDataPointer<SurfaceInteraction>(
                params.traversable_handle,
                ray_ctx.ray.origin,
                ray_ctx.ray.direction,
                params.scene_epsilon,                   // tmin: Start ray at ray_origin + tmin * ray_direction
                std::numeric_limits<float>::infinity(), // tmax: End ray at ray_origin + tmax * ray_direction
                params.surface_interaction_trace_params,
                &si
        );

        // Terminate this ray
        if (!si.is_finite() || si.bsdf == nullptr)
            continue;

        // Direct illumination for each emitter
        for (uint32_t emitter_index = 0; emitter_index < params.emitters.count; ++emitter_index)
        {
            const EmitterVPtrTable *emitter = params.emitters[emitter_index];
            const BSDFVPtrTable *bsdf = si.bsdf;

            // Generate ray towards light source
            EmitterSamplingResult emitter_sampling_result = emitter->sampleLight(si, dummy_rng);

            // Cast shadow ray
            bool occluded = traceOcclusion(
                    params.traversable_handle,
                    si.position,
                    emitter_sampling_result.direction_to_light,
                    params.scene_epsilon,
                    emitter_sampling_result.distance_to_light - params.scene_epsilon,
                    params.occlusion_trace_params
            );
            if (occluded)
                continue;

            // Evaluate bsdf
            BSDFEvalResult bsdf_result = bsdf->evalBSDF(si, emitter_sampling_result.direction_to_light, +BSDFComponentFlag::Any);

            // Accumulate output radiance stored in pixel
            output_radiance += ray_ctx.throughput * bsdf_result.bsdf_value * emitter_sampling_result.radiance_weight_at_receiver;
        }

        // No further indirect bounces for this surface interaction if ray depth is exceeded
        if (ray_ctx.depth >= MAX_TRACE_DEPTH)
            continue;

        // Indirect illumination, generate next ray
        if (has_flag(si.bsdf->component_flags, BSDFComponentFlag::IdealReflection) && !stack_of_rays_to_trace.full())
        {
            BSDFSamplingResult bsdf_sampling_result = si.bsdf->sampleBSDF(si, +BSDFComponentFlag::IdealReflection, dummy_rng);
            if (bsdf_sampling_result.sampling_pdf != 0)
            {
                RayContext new_ray_ctx;
                new_ray_ctx.ray.origin = si.position;
                new_ray_ctx.ray.direction = bsdf_sampling_result.outgoing_ray_dir;
                new_ray_ctx.throughput = ray_ctx.throughput * bsdf_sampling_result.bsdf_weight;
                new_ray_ctx.depth = ray_ctx.depth + 1;
                stack_of_rays_to_trace.push(new_ray_ctx);
            }
        }
        if (has_flag(si.bsdf->component_flags, BSDFComponentFlag::IdealTransmission) && !stack_of_rays_to_trace.full())
        {
            BSDFSamplingResult bsdf_sampling_result = si.bsdf->sampleBSDF(si, +BSDFComponentFlag::IdealTransmission, dummy_rng);
            if (bsdf_sampling_result.sampling_pdf != 0)
            {
                RayContext new_ray_ctx;
                new_ray_ctx.ray.origin = si.position;
                new_ray_ctx.ray.direction = bsdf_sampling_result.outgoing_ray_dir;
                new_ray_ctx.throughput = ray_ctx.throughput * bsdf_sampling_result.bsdf_weight;
                new_ray_ctx.depth = ray_ctx.depth + 1;
                stack_of_rays_to_trace.push(new_ray_ctx);
            }
        }
    }

    // Write linear output color
    params.output_radiance(pixel_index).value() = output_radiance;
}
