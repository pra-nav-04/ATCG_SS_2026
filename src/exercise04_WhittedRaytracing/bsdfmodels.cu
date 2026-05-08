#include "bsdfmodels.cuh"

#include "opg/scene/utility/interaction.cuh"

#include <optix.h>

// Schlick's approximation to the fresnel reflectance term
// See https://en.wikipedia.org/wiki/Schlick%27s_approximation
__device__ float fresnel_schlick( const float F0, const float VdotH )
{
    return F0 + ( 1.0f - F0 ) * glm::pow( glm::max(0.0f, 1.0f - VdotH), 5.0f );
}

__device__ glm::vec3 fresnel_schlick( const glm::vec3 F0, const float VdotH )
{
    return F0 + ( glm::vec3(1.0f) - F0 ) * glm::pow( glm::max(0.0f, 1.0f - VdotH), 5.0f );
}


extern "C" __device__ BSDFEvalResult __direct_callable__opaque_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const OpaqueBSDFData *sbt_data = *reinterpret_cast<const OpaqueBSDFData **>(optixGetSbtDataPointer());

    float NdotV = glm::dot(si.normal, -si.incoming_ray_dir); // incoming_ray_dir points towards surface
    float NdotL = glm::dot(si.normal, outgoing_ray_dir);

    // if (sign(NdotL) == sign(NdotV))
    //    clampedNdotL = abs(NdotL);
    // else
    //    clampedNdotL = 0;
    float clampedNdotL = glm::max(0.0f, NdotL * glm::sign(NdotV));

    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf * clampedNdotL;

    BSDFEvalResult result;
    result.bsdf_value = diffuse_bsdf;
    result.sampling_pdf = 0; // No diffuse BSDF importance sampling
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__opaque_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
{
    const OpaqueBSDFData *sbt_data = *reinterpret_cast<const OpaqueBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample

    // Check if there is no specular component present
    if (!has_flag(component_flags, BSDFComponentFlag::IdealReflection))
        return result;
    // Check if the specular component is zero
    if (glm::dot(sbt_data->specular_F0, sbt_data->specular_F0) < 1e-6)
        return result;

    /* Implement:
     * - Specular reflections on opaque materials (BRDF of a specular reflection with given reflectance at normal incidence).
     *   - Compute the outgoing ray direction
     *   - Compute the BSDF for the reflection of the incoming ray direction to the outgoing ray direction.
     *   - Set the sampling pdf to 1 to indicate a valid result (The sampling pdf is used later for stochastic sampling methods)
     */
    result.outgoing_ray_dir = glm::normalize(glm::reflect(si.incoming_ray_dir, si.normal));
    result.bsdf_weight = sbt_data->specular_F0;
    result.sampling_pdf = 1.0f;

    return result;
}


extern "C" __device__ BSDFEvalResult __direct_callable__refractive_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    // No direct illumination on refractive materials!
    BSDFEvalResult result;
    result.bsdf_value = glm::vec3(0);
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__refractive_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
{
    const RefractiveBSDFData *sbt_data = *reinterpret_cast<const RefractiveBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample

    /* Implement:
     * - Reflections and transmissions on refractive materials.
     *   - Compute the outgoing ray direction.
     *     Hint: Check for `component_flags == +BSDFComponentFlag::IdealReflection` or `component_flags == +BSDFComponentFlag::IdealTransmission`
     *           to determine if a reflection or transmission ray should be generated.
     *           The `+` is neccessary to convert from the `enum` type to `uint32_t`...
     *   - Compute the BSDF for the reflection of the incoming ray direction to the outgoing ray direction.
     *   - Set the sampling pdf to 1 to indicate a valid result (The sampling pdf is used later for stochastic sampling methods).
     *   Hint: The surface normals point outwards.
     *   Hint: You can use Schlick's approximation for the Fresnel term to compute the amount of light reflected or transmitted.
     */
    const bool entering = glm::dot(si.incoming_ray_dir, si.normal) < 0.0f;
    const glm::vec3 oriented_normal = entering ? si.normal : -si.normal;
    const float n1 = entering ? 1.0f : sbt_data->index_of_refraction;
    const float n2 = entering ? sbt_data->index_of_refraction : 1.0f;
    const float eta = n1 / n2;
    const float cos_i = glm::max(0.0f, glm::dot(oriented_normal, -si.incoming_ray_dir));
    const float r0 = (n1 - n2) / (n1 + n2);
    const float fresnel = fresnel_schlick(r0 * r0, cos_i);

    if (component_flags == +BSDFComponentFlag::IdealReflection)
    {
        result.outgoing_ray_dir = glm::normalize(glm::reflect(si.incoming_ray_dir, oriented_normal));
        result.bsdf_weight = glm::vec3(fresnel);
        result.sampling_pdf = 1.0f;
        return result;
    }

    if (component_flags == +BSDFComponentFlag::IdealTransmission)
    {
        glm::vec3 refracted_dir = glm::refract(si.incoming_ray_dir, oriented_normal, eta);
        if (glm::length2(refracted_dir) == 0.0f)
            return result;

        result.outgoing_ray_dir = glm::normalize(refracted_dir);
        result.bsdf_weight = glm::vec3(1.0f - fresnel);
        result.sampling_pdf = 1.0f;
    }

    return result;
}
