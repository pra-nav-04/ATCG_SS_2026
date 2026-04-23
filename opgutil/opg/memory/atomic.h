#pragma once

#include "opg/glmwrapper.h"

#ifdef __CUDACC__

// Implementation of atomicMax function for float
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Implementation of atomicMin function for float
__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template<int N, typename T, glm::qualifier Q>
__device__ OPG_INLINE glm::vec<N, T, Q> atomicAdd(glm::vec<N, T, Q> *address, glm::vec<N, T, Q> val)
{
    // Perform component-wise atomic add.
    glm::vec<N, T, Q> result;
    for (int i = 0; i < N; ++i)
        result[i] = atomicAdd(&(*address)[i], val[i]);
    return result;
}

template<int N, typename T, glm::qualifier Q>
__device__ OPG_INLINE glm::vec<N, T, Q> atomicMax(glm::vec<N, T, Q> *address, glm::vec<N, T, Q> val)
{
    // Perform component-wise atomic max.
    glm::vec<N, T, Q> result;
    for (int i = 0; i < N; ++i)
        result[i] = atomicMax(&(*address)[i], val[i]);
    return result;
}

template<int N, typename T, glm::qualifier Q>
__device__ OPG_INLINE glm::vec<N, T, Q> atomicMin(glm::vec<N, T, Q> *address, glm::vec<N, T, Q> val)
{
    // Perform component-wise atomic min.
    glm::vec<N, T, Q> result;
    for (int i = 0; i < N; ++i)
        result[i] = atomicMin(&(*address)[i], val[i]);
    return result;
}

#endif // __CUDACC__
