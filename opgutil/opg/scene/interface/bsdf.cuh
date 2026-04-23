#pragma once

#include "opg/glmwrapper.h"
#include "opg/hostdevice/random.h"
#include "opg/enum.h"

#include <optix.h>


// Forward declaration...
struct SurfaceInteraction;


// The BSDF component flags are used to give the caller of a BSDF some
// information on what kind of BDSF it is.
// The Ideal* flags indicate perfect mirror reflections/transmissions.
// The Diffuse* flags indicate that the BSDF can be considered as diffuse.
// The Glossy* flags are used for specular BSDFs that are usually seen on surfaces with some roughness.
enum class BSDFComponentFlag : uint32_t
{
    None                = 0x00,
    IdealReflection     = 0x01,
    GlossyReflection    = 0x02,
    DiffuseReflection   = 0x04,
    IdealTransmission   = 0x08,
    GlossyTransmission  = 0x10,
    DiffuseTransmission = 0x20,

    // TODO Note: these are collections of "types"!
    AnyDelta            = IdealReflection | IdealTransmission,
    AnyReflection       = IdealReflection | GlossyReflection | DiffuseReflection,
    AnyTransmission     = IdealTransmission | GlossyTransmission | DiffuseTransmission,
    Any                 = AnyReflection | AnyTransmission,
};
OPG_DECLARE_ENUM_OPERATIONS(BSDFComponentFlag);
typedef uint32_t BSDFComponentFlags;

// This struct encapsulates all information that is generated when sampling a new ray direction from a BSDF
struct BSDFSamplingResult
{
    // The sampled outgoing ray direction.
    glm::vec3 outgoing_ray_dir;
    // BSDF value divided by sampling PDF for the sampled direction.
    glm::vec3 bsdf_weight;
    // The sampling PDF is useful for multiple importance sampling.
    float     sampling_pdf;

    // Construct an empty sampling result, indicating that the sampling failed.
    OPG_HOSTDEVICE static BSDFSamplingResult empty()
    {
        BSDFSamplingResult result;
        result.outgoing_ray_dir = glm::vec3(0);
        result.bsdf_weight = glm::vec3(0);
        result.sampling_pdf = 0;
        return result;
    }
};

struct BSDFEvalResult
{
    // BSDF value for the given direction. NOTE: This is not divided by the BSDF sampling PDF!
    glm::vec3 bsdf_value;
    // The probability of sampling the given direction via BSDF importance sampling.
    // The sampling PDF is useful for multiple importance sampling.
    float     sampling_pdf;

    // Construct an empty sampling result, indicating that the sampling failed.
    OPG_HOSTDEVICE static BSDFEvalResult empty()
    {
        BSDFEvalResult result;
        result.bsdf_value = glm::vec3(0);
        result.sampling_pdf = 0;
        return result;
    }
};

struct BSDFVPtrTable
{
    BSDFComponentFlags component_flags;

    // Indices of callable programs in shader binding table
    uint32_t evalCallIndex              OPG_MEMBER_INIT(~0u);
    uint32_t sampleCallIndex            OPG_MEMBER_INIT(~0u);
    uint32_t evalDiffuseAlbedoIndex     OPG_MEMBER_INIT(~0u);


    //
    // The following code is only valid in device code
    //
    #ifdef __CUDACC__

    // Evaluate the BSDF (including the <N,L> term) at a surface interaction for a given outgoing ray direction.
    // Optionally computes the probability of sampling the given direction using the `sampleBSDF` method via BSDF importance sampling, which is used for multiple-importance sampling.
    __device__ BSDFEvalResult evalBSDF(const SurfaceInteraction &si, const glm::vec3 &outgoing_ray_dir, BSDFComponentFlags component_flags) const
    {
        return optixDirectCall<BSDFEvalResult, const SurfaceInteraction &, const glm::vec3 &, BSDFComponentFlags>(evalCallIndex, si, outgoing_ray_dir, component_flags);
    }

    // Generate a new ray direction from this BSDF, given a mask of component flags that should be used here.
    // This is used for importance sampling when we do path tracing.
    __device__ BSDFSamplingResult sampleBSDF(const SurfaceInteraction &si, BSDFComponentFlags component_flags, PCG32 &rng) const
    {
        return optixDirectCall<BSDFSamplingResult, const SurfaceInteraction &, BSDFComponentFlags, PCG32 &>(sampleCallIndex, si, component_flags, rng);
    }

    // Evaluate the diffuse albedo at a surface interaction without knowing the incoming light direction.
    __device__ glm::vec3 evalDiffuseAlbedo(const SurfaceInteraction &si) const
    {
        return optixDirectCall<glm::vec3, const SurfaceInteraction &>(evalDiffuseAlbedoIndex, si);
    }

    #endif // __CUDACC__
};
