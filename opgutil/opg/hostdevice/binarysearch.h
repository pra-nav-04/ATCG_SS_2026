#pragma once

#include <cstdint>
#include "opg/preprocessor.h"

namespace opg {

template <typename T>
OPG_INLINE OPG_HOSTDEVICE uint32_t binary_search(BufferView<T> sorted_array, T value)
{
    // Find first element in sorted_array that is larger than value.
    uint32_t left = 0;
    uint32_t right = sorted_array.count-1;
    while (left < right)
    {
        uint32_t mid = (left + right) / 2;
        if (sorted_array[mid] < value)
            left = mid+1;
        else
            right = mid;
    }
    return left;
}

} // namespace opg
