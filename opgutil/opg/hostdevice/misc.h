#pragma once

#include "opg/preprocessor.h"

template <typename T>
OPG_INLINE OPG_HOSTDEVICE T ceil_to_multiple(T value, T mod)
{
    return value-T(1) + mod - ((value-T(1)) % mod);
}

template <typename T>
OPG_INLINE OPG_HOSTDEVICE T ceil_div(T num, T den)
{
    return (num-T(1)) / den + T(1);
}
