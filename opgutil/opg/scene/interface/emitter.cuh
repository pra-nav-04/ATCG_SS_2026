#pragma once

#include <optix.h>

#include "opg/glmwrapper.h"
#include "opg/hostdevice/random.h"
#include "opg/enum.h"


// Forward declaration
struct Interaction;
struct SurfaceInteraction;


// The Emitter flags are used to give the caller of an Emitter some
// information on what kind of Emitter it is.
enum class EmitterFlag : uint32_t
{
    // This emitter is infinitely far away from the scene
    DistantEmitter      = 0x1,
    // This emitter has no surface area (i.e. point light) and can therefore not be encountered by regular ray tracing
    InfinitesimalSize   = 0x2
};
OPG_DECLARE_ENUM_OPERATIONS(EmitterFlag);
typedef uint32_t EmitterFlags;

// This struct encapsulates all information that is generated when sampling a new ray direction from a surface to an Emitter
struct EmitterSamplingResult
{
    glm::vec3 direction_to_light;
    float     distance_to_light;
    // The surface normal of the emitter, if the emitter has any.
    // For point light sources or distant emitters this is set to nan!
    glm::vec3 normal_at_light;
    // Radiance emitted by the light source divided by the sampling pdf.
    glm::vec3 radiance_weight_at_receiver;
    // The sampling PDF is useful for multiple importance sampling.
    float     sampling_pdf;
};

// This structure encapsulates all information that is generated when sampling a random photon on the surface of an Emitter
struct EmitterPhotonSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    // The surface normal of the emitter, if the emitter has any.
    // For point light sources or distant emitters this is set to nan!
    glm::vec3 normal_at_light;
    // Radiance emitted by the light source divided by the sampling pdf.
    glm::vec3 radiance_weight;
    // The sampling PDF is useful for multiple importance sampling.
    float sampling_pdf;
};

struct EmitterVPtrTable
{
    EmitterFlags flags;
    // Weight for selecting this emitter during next event estimation.
    uint32_t emitter_weight;

    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;
    uint32_t evalSamplingPdfCallIndex;
    uint32_t samplePhotonCallIndex;

    //
    // The following code is only valid in device code
    //
    #ifdef __CUDACC__

    // Evaluate the emitter at a position on the emitter surface.
    __device__ glm::vec3 evalLight(const SurfaceInteraction &si) const
    {
        return optixDirectCall<glm::vec3, const SurfaceInteraction &>(evalCallIndex, si);
    }

    // Given some surface or medium interaction in the scene, generate a new ray direction towards this Emitter.
    __device__ EmitterSamplingResult sampleLight(const Interaction &si, PCG32 &rng) const
    {
        return optixDirectCall<EmitterSamplingResult, const Interaction &, PCG32 &>(sampleCallIndex, si, rng);
    }

    // Given a surface or medium interaction and a sampled ray direction, what is the probability that this direction was generated using the `sampleLight` method.
    // This is used for multiple-importance sampling.
    __device__ float evalLightSamplingPdf(const Interaction &si, const SurfaceInteraction &si_on_light) const
    {
        return optixDirectCall<float, const Interaction &, const SurfaceInteraction &>(evalSamplingPdfCallIndex, si, si_on_light);
    }

    // Generate a random photon on the surface of this light source
    __device__ EmitterPhotonSamplingResult samplePhoton(PCG32 &rng) const
    {
        return optixDirectCall<EmitterPhotonSamplingResult, PCG32 &>(samplePhotonCallIndex, rng);
    }

    #endif // __CUDACC__
};
