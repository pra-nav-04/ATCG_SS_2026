#include "bsdfmodels.cuh"

#include "opg/scene/utility/interaction.cuh"

#include <optix.h>

// Schlick's approximation to the fresnel reflectance term
// See https://en.wikipedia.org/wiki/Schlick%27s_approximation
__forceinline__ __device__ float fresnel_schlick( const float F0, const float VdotH )
{
    return F0 + ( 1.0f - F0 ) * glm::pow( glm::max(0.0f, 1.0f - VdotH), 5.0f );
}

__forceinline__ __device__ glm::vec3 fresnel_schlick( const glm::vec3 F0, const float VdotH )
{
    return F0 + ( glm::vec3(1.0f) - F0 ) * glm::pow( glm::max(0.0f, 1.0f - VdotH), 5.0f );
}


extern "C" __device__ BSDFEvalResult __direct_callable__opaque_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const OpaqueBSDFData *sbt_data = *reinterpret_cast<const OpaqueBSDFData **>(optixGetSbtDataPointer());
    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf * glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = diffuse_bsdf;
    result.sampling_pdf = 0;
    return result;
}

extern "C" __device__ BSDFSamplingResult __direct_callable__opaque_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
{
    const OpaqueBSDFData *sbt_data = *reinterpret_cast<const OpaqueBSDFData **>(optixGetSbtDataPointer());

    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample

    if (!has_flag(component_flags, BSDFComponentFlag::IdealReflection))
        return result;
    if (glm::dot(sbt_data->specular_F0, sbt_data->specular_F0) < 1e-6)
        return result;

    result.outgoing_ray_dir = glm::reflect(si.incoming_ray_dir, si.normal);
    result.bsdf_weight = sbt_data->specular_F0; // TODO evaluate Schlick's Fresnel!
    result.sampling_pdf = 1;

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
    result.sampling_pdf = 0;

    bool outsidein = glm::dot(si.incoming_ray_dir, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta = outsidein ? 1.0f / sbt_data->index_of_refraction : sbt_data->index_of_refraction;

    glm::vec3 transmitted_ray_dir = glm::refract(si.incoming_ray_dir, interface_normal, eta);
    glm::vec3 reflected_ray_dir = glm::reflect(si.incoming_ray_dir, interface_normal);

    float F0 = (eta - 1) / (eta + 1);
    F0 = F0 * F0;

    float NdotL = glm::abs(glm::dot(si.incoming_ray_dir, interface_normal));

    float reflection_probability = fresnel_schlick(F0, NdotL);
    float transmission_probability = 1.0f - reflection_probability;

    if (glm::dot(transmitted_ray_dir, transmitted_ray_dir) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = 0.0f;
        reflection_probability = 1.0f;
    }

    if (component_flags == +BSDFComponentFlag::IdealReflection && reflection_probability > 0)
    {
        result.bsdf_weight = glm::vec3(reflection_probability);
        result.outgoing_ray_dir = reflected_ray_dir;
        result.sampling_pdf = 1;
    }
    else if (component_flags == +BSDFComponentFlag::IdealTransmission && transmission_probability > 0)
    {
        result.bsdf_weight = glm::vec3(transmission_probability);
        result.outgoing_ray_dir = transmitted_ray_dir;
        result.sampling_pdf = 1;
    }

    return result;
}



// 



//
// Phong BSDF
//

extern "C" __device__ BSDFEvalResult __direct_callable__phong_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const PhongBSDFData *sbt_data = *reinterpret_cast<const PhongBSDFData **>(optixGetSbtDataPointer());

    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf;
    glm::vec3 specular_bsdf = glm::vec3(0);

    /* Implement:
     * Phong / Cosine-Lobe BRDF (energy-conserving normalisation)
     *     f_s = F0 * (n + 2) / (2 pi) * max(0, R . V)^n
     * with R = reflect(-L, N), n = exponent.
     */

    {
        glm::vec3 N = si.normal;
        if (glm::dot(si.incoming_ray_dir, N) > 0.0f)
            N = -N;
        const glm::vec3 V = -si.incoming_ray_dir;
        const glm::vec3 L = outgoing_ray_dir;

        if (glm::dot(N, L) > 0.0f && glm::dot(N, V) > 0.0f)
        {
            const glm::vec3 R = glm::reflect(-L, N);
            const float RdotV = glm::max(0.0f, glm::dot(R, V));
            const float n = sbt_data->exponent;
            specular_bsdf = sbt_data->specular_F0 * ((n + 2.0f) / (2.0f * M_PIf)) * glm::pow(RdotV, n);
        }
    }

    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * clampedNdotL;
    result.sampling_pdf = 0; // Importance sampling not supported in this exercise.
    return result;
}


//
// Ward BSDF
//

extern "C" __device__ BSDFEvalResult __direct_callable__ward_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const WardBSDFData *sbt_data = *reinterpret_cast<const WardBSDFData **>(optixGetSbtDataPointer());


    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf;

    glm::vec3 specular_bsdf = glm::vec3(0);

    /* Implement:
     * Anisotropic Geisler-Moroder Ward BRDF using the unnormalised
     * halfway vector H = L + V (slide 22 of the Geisler-Moroder slides):
     *
     *   f_s = F * |H|^2 / (pi * a_t * a_b * (N.L)^2 * (N.V)^2)
     *           * exp(- ((T.H)^2/a_t^2 + (B.H)^2/a_b^2) / (N.H)^2)
     *
     * with F = Schlick(F0, V . H_hat).
     */

    {
        glm::vec3 N = si.normal;
        if (glm::dot(si.incoming_ray_dir, N) > 0.0f)
            N = -N;
        glm::vec3 T = si.tangent - N * glm::dot(N, si.tangent);
        if (glm::dot(T, T) < 1e-12f)
            T = glm::vec3(0);
        else
            T = glm::normalize(T);
        const glm::vec3 B = glm::cross(N, T);

        const glm::vec3 V = -si.incoming_ray_dir;
        const glm::vec3 L = outgoing_ray_dir;

        const float NdotL = glm::dot(N, L);
        const float NdotV = glm::dot(N, V);

        if (NdotL > 0.0f && NdotV > 0.0f)
        {
            const glm::vec3 H = L + V;
            const float HdotH = glm::dot(H, H);
            const float NdotH = glm::dot(N, H);
            const float TdotH = glm::dot(T, H);
            const float BdotH = glm::dot(B, H);

            const float at = glm::max(1e-4f, sbt_data->roughness_tangent);
            const float ab = glm::max(1e-4f, sbt_data->roughness_bitangent);

            if (NdotH > 0.0f && HdotH > 0.0f)
            {
                const float exp_arg = -((TdotH * TdotH) / (at * at) + (BdotH * BdotH) / (ab * ab)) / (NdotH * NdotH);
                const float D_term = HdotH * glm::exp(exp_arg)
                                   / (M_PIf * at * ab * (NdotL * NdotL) * (NdotV * NdotV));

                const glm::vec3 Hn = H * (1.0f / glm::sqrt(HdotH));
                const float VdotHn = glm::max(0.0f, glm::dot(V, Hn));
                const glm::vec3 F = fresnel_schlick(sbt_data->specular_F0, VdotHn);

                specular_bsdf = F * D_term;
            }
        }
    }

    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * clampedNdotL;
    result.sampling_pdf = 0; // Importance sampling not supported in this exercise.
    return result;
}


//
// GGX BSDF
//

extern "C" __device__ BSDFEvalResult __direct_callable__ggx_evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags)
{
    const GGXBSDFData *sbt_data = *reinterpret_cast<const GGXBSDFData **>(optixGetSbtDataPointer());

    glm::vec3 diffuse_bsdf = sbt_data->diffuse_color / M_PIf;

    glm::vec3 specular_bsdf = glm::vec3(0);

    /* Implement:
     * Anisotropic Cook-Torrance with GGX NDF, Smith G, Schlick F.
     *
     *   D(h)   = 1 / ( pi a_t a_b ( (T.h)^2/a_t^2 + (B.h)^2/a_b^2 + (N.h)^2 )^2 )
     *   Lambda(w) = (-1 + sqrt(1 + a^2 tan^2 theta_w)) / 2,
     *               with a^2 tan^2 theta = (a_t^2 (T.w)^2 + a_b^2 (B.w)^2) / (N.w)^2
     *   G1(w)  = 1 / (1 + Lambda(w))
     *   G      = G1(L) * G1(V)
     *   F      = Schlick(F0, V.H)
     *   f_s    = D G F / (4 (N.L)(N.V))
     */

    {
        glm::vec3 N = si.normal;
        if (glm::dot(si.incoming_ray_dir, N) > 0.0f)
            N = -N;
        glm::vec3 T = si.tangent - N * glm::dot(N, si.tangent);
        if (glm::dot(T, T) < 1e-12f)
            T = glm::vec3(0);
        else
            T = glm::normalize(T);
        const glm::vec3 B = glm::cross(N, T);

        const glm::vec3 V = -si.incoming_ray_dir;
        const glm::vec3 L = outgoing_ray_dir;

        const float NdotL = glm::dot(N, L);
        const float NdotV = glm::dot(N, V);

        if (NdotL > 0.0f && NdotV > 0.0f)
        {
            const glm::vec3 Hn = glm::normalize(L + V);
            const float NdotH = glm::dot(N, Hn);
            const float TdotH = glm::dot(T, Hn);
            const float BdotH = glm::dot(B, Hn);
            const float VdotH = glm::max(0.0f, glm::dot(V, Hn));

            const float at = glm::max(1e-4f, sbt_data->roughness_tangent);
            const float ab = glm::max(1e-4f, sbt_data->roughness_bitangent);

            // Anisotropic GGX NDF
            const float denom_D = (TdotH * TdotH) / (at * at)
                                + (BdotH * BdotH) / (ab * ab)
                                + (NdotH * NdotH);
            const float D = 1.0f / (M_PIf * at * ab * denom_D * denom_D);

            // Smith Lambda for anisotropic GGX
            auto lambda = [&](const glm::vec3 &w) {
                const float NdotW = glm::dot(N, w);
                const float TdotW = glm::dot(T, w);
                const float BdotW = glm::dot(B, w);
                const float a2tan2 = (at * at * TdotW * TdotW + ab * ab * BdotW * BdotW) / (NdotW * NdotW);
                return 0.5f * (-1.0f + glm::sqrt(1.0f + a2tan2));
            };
            const float G = 1.0f / ((1.0f + lambda(L)) * (1.0f + lambda(V)));

            const glm::vec3 F = fresnel_schlick(sbt_data->specular_F0, VdotH);

            specular_bsdf = (D * G) * F / (4.0f * NdotL * NdotV);
        }
    }

    float clampedNdotL = glm::max(0.0f, glm::dot(outgoing_ray_dir, si.normal) * -glm::sign(glm::dot(si.incoming_ray_dir, si.normal)));

    BSDFEvalResult result;
    result.bsdf_value = (diffuse_bsdf + specular_bsdf) * clampedNdotL;
    result.sampling_pdf = 0; // Importance sampling not supported in this exercise.
    return result;
}



// Shared dummy BSDF sampling method
extern "C" __device__ BSDFSamplingResult __direct_callable__phong_ward_ggx_sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &unused_rng)
{
    BSDFSamplingResult result;
    result.sampling_pdf = 0; // invalid sample

    // Importance sampling of glossy BSDFs is added in a future exercise...
    // For now, there is no importance sampling support for this BSDF
    return result;
}
