#pragma once

#include <stdint.h>

#include "opg/preprocessor.h"

namespace opg {

// Simple stack datastructure that stores its content on the stack.
template <typename T, uint32_t Capacity>
class Stack
{
public:
    OPG_INLINE OPG_HOSTDEVICE uint32_t  size()     const { return m_size; }
    OPG_INLINE OPG_HOSTDEVICE uint32_t  capacity() const { return Capacity; }
    OPG_INLINE OPG_HOSTDEVICE bool      empty()    const { return m_size == 0; }
    OPG_INLINE OPG_HOSTDEVICE bool      full()     const { return m_size == Capacity; }

    OPG_INLINE OPG_HOSTDEVICE T         peek()     const { return empty() ? T() : m_data[m_size-1]; }
    OPG_INLINE OPG_HOSTDEVICE T         pop()            { return empty() ? T() : m_data[--m_size]; }
    OPG_INLINE OPG_HOSTDEVICE void      push(T val)      { if (!full()) m_data[m_size++] = val; }

private:
    T           m_data[Capacity];
    uint32_t    m_size = 0;
};

} // end namespace opg
