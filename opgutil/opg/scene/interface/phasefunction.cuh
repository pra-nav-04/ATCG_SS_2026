#pragma once

#include "opg/glmwrapper.h"
#include "opg/hostdevice/random.h"

#include <optix.h>


// Forward declaration...
struct MediumInteraction;


struct PhaseFunctionSamplingResult
{
    // The sampled outgoing ray direction.
    glm::vec3 outgoing_ray_dir;
    // Phase function value divided by sampling PDF for the sampled direction.
    glm::vec3 phase_function_weight;
    // The sampling PDF is useful for multiple importance sampling.
    float     sampling_pdf;
};

struct PhaseFunctionEvalResult
{
    // Phase function value for the given direction. NOTE: This is not divided by the sampling PDF!
    glm::vec3 phase_function_value;
    // The probability of sampling the given direction via phase-function importance sampling.
    // The sampling PDF is useful for multiple importance sampling.
    float     sampling_pdf;
};

struct PhaseFunctionVPtrTable
{
    // Indices of callable programs in shader binding table
    uint32_t evalCallIndex              OPG_MEMBER_INIT(~0u);
    uint32_t sampleCallIndex            OPG_MEMBER_INIT(~0u);

    //
    // The following code is only valid in device code
    //
    #ifdef __CUDACC__

    __device__ PhaseFunctionEvalResult evalPhaseFunction(const MediumInteraction &interaction, const glm::vec3 &outgoing_ray_dir) const
    {
        return optixDirectCall<PhaseFunctionEvalResult, const MediumInteraction &, const glm::vec3 &>(evalCallIndex, interaction, outgoing_ray_dir);
    }

    __device__ PhaseFunctionSamplingResult samplePhaseFunction(const MediumInteraction &interaction, PCG32 &rng) const
    {
        return optixDirectCall<PhaseFunctionSamplingResult, const MediumInteraction &, PCG32 &>(sampleCallIndex, interaction, rng);
    }

    #endif // __CUDACC__
};
