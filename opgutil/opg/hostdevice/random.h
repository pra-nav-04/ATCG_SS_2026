#pragma once

#include "opg/preprocessor.h"
#include "opg/glmwrapper.h"

// Pseudorandom number generation using the Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
static OPG_INLINE OPG_HOSTDEVICE uint32_t sampleTEA32(uint32_t val0, uint32_t val1, uint32_t rounds = 4)
{
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;

    for (uint32_t n = 0; n < rounds; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }

    return v0;
}

// Pseudorandom number generation using the Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
static OPG_INLINE OPG_HOSTDEVICE uint64_t sampleTEA64(uint32_t val0, uint32_t val1, uint32_t rounds = 4)
{
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;

    for (uint32_t n = 0; n < rounds; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }

    return static_cast<uint64_t>(v0) | (static_cast<uint64_t>(v1) << 32);
}


constexpr uint64_t PCG32_DEFAULT_STATE  = 0x853c49e6748fea9bULL;
constexpr uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
constexpr uint64_t PCG32_MULT           = 0x5851f42d4c957f2dULL;

/// PCG32 pseudorandom number generator proposed by Melissa O'Neill
struct PCG32
{
    OPG_INLINE OPG_HOSTDEVICE PCG32(uint64_t initstate = PCG32_DEFAULT_STATE, uint64_t initseq = PCG32_DEFAULT_STREAM)
    {
        seed(initstate, initseq);
    }

    OPG_INLINE OPG_HOSTDEVICE void seed(uint64_t initstate, uint64_t initseq)
    {
        state = 0;
        inc = (initseq << 1) | 1;
        nextUint32();
        state += initstate;
        nextUint32();
    }

    OPG_INLINE OPG_HOSTDEVICE uint32_t nextUint32()
    {
        uint64_t oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18) ^ oldstate) >> 27);
        uint32_t rot_offset = static_cast<uint32_t>(oldstate >> 59);
        return (xorshifted >> rot_offset) | (xorshifted << (32-rot_offset));
    }

    OPG_INLINE OPG_HOSTDEVICE uint64_t nextUint64()
    {
        return static_cast<uint64_t>(nextUint32()) | (static_cast<uint64_t>(nextUint32()) << 32);
    }

    // Generate a random float in (0, 1) containing 23 bits of randomness
    OPG_INLINE OPG_HOSTDEVICE float nextFloat()
    {
        // First generate a random number in (1, 2) and subtract 1.
        uint32_t rand_bits = nextUint32();
        uint32_t mantissa_bits = (rand_bits & (~0u << 9)) != 0u ? rand_bits >> 9 : ((rand_bits << 14) | (1u << 13));
        uint32_t float_bits = (mantissa_bits) | 0x3f800000u;
        return reinterpret_cast<float&>(float_bits) - 1.0f;
    }

    // Generate a random double in (0, 1) containing 32 bits of randomness (lower mantissa bits are set to 0 here)
    OPG_INLINE OPG_HOSTDEVICE double nextDouble()
    {
        // First generate a random number in (1, 2) and subtract 1
        uint64_t bits = (static_cast<uint64_t>(nextUint32()) << 20) | (1ull << 19) | 0x3ff0000000000000ull;
        return reinterpret_cast<double&>(bits) - 1.0;
    }


    // Generate a 1d sample in (0, 1)
    OPG_INLINE OPG_HOSTDEVICE float next1d()
    {
        return nextFloat();
    }

    // Generate a 2d sample in (0, 1)^2
    OPG_INLINE OPG_HOSTDEVICE glm::vec2 next2d()
    {
        return glm::vec2(nextFloat(), nextFloat());
    }

    // Generate a 3d sample in (0, 1)^3
    OPG_INLINE OPG_HOSTDEVICE glm::vec3 next3d()
    {
        return glm::vec3(nextFloat(), nextFloat(), nextFloat());
    }

    // Generate a 4d sample in (0, 1)^4
    OPG_INLINE OPG_HOSTDEVICE glm::vec4 next4d()
    {
        return glm::vec4(nextFloat(), nextFloat(), nextFloat(), nextFloat());
    }


    uint64_t state;  // RNG state.  All values are possible.
    uint64_t inc;    // Controls which RNG sequence (stream) is selected. Must *always* be odd.
};
