#pragma once

#include "opg/glmwrapper.h"
#include "opg/hostdevice/random.h"
#include "opg/hostdevice/ray.h"

#include <optix.h>

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/interface/phasefunction.cuh"


struct MediumSamplingResult
{
    // The resulting interaction with the medium.
    MediumInteraction interaction;
    // Transmittance divided by the sampling pdf.
    glm::vec3   transmittance_weight;
};

struct MediumVPtrTable
{
    // The phase function of this medium
    const PhaseFunctionVPtrTable *phase_function;

    // Indices of callable programs in shader binding table
    uint32_t evalCallIndex              OPG_MEMBER_INIT(~0u);
    uint32_t sampleCallIndex            OPG_MEMBER_INIT(~0u);

    //
    // The following code is only valid in device code
    //
    #ifdef __CUDACC__

    // Evaluate the transmittance over a certain distance inside of this medium starting at the (medium) interaction.
    __device__ glm::vec3 evalTransmittance(const opg::Ray &ray, float distance, PCG32 &rng) const
    {
        return optixDirectCall<glm::vec3, const opg::Ray &, float, PCG32 &>(evalCallIndex, ray, distance, rng);
    }

    // Sample the position of a new medium event starting at the given (medium) interaction.
    __device__ MediumSamplingResult sampleMediumEvent(const opg::Ray &ray, float max_distance, PCG32 &rng) const
    {
        return optixDirectCall<MediumSamplingResult, const opg::Ray &, float, PCG32 &>(sampleCallIndex, ray, max_distance, rng);
    }

    #endif // __CUDACC__
};
