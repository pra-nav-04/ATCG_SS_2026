#pragma once

#include <stdint.h>

#include "opg/preprocessor.h"

namespace opg {

// Simple std::vector-like datastructure that stores its content on the stack.
template <typename T, uint32_t Capacity>
class Vector
{
public:
    OPG_INLINE OPG_HOSTDEVICE uint32_t  size()     const { return m_size; }
    OPG_INLINE OPG_HOSTDEVICE uint32_t  capacity() const { return Capacity; }
    OPG_INLINE OPG_HOSTDEVICE bool      empty()    const { return m_size == 0; }
    OPG_INLINE OPG_HOSTDEVICE bool      full()     const { return m_size == Capacity; }

    OPG_INLINE OPG_HOSTDEVICE T&        back()           { return m_data[m_size-1]; }
    OPG_INLINE OPG_HOSTDEVICE const T&  back()     const { return m_data[m_size-1]; }
    OPG_INLINE OPG_HOSTDEVICE void      push_back(T val) { if (!full()) m_data[m_size++] = val; }
    OPG_INLINE OPG_HOSTDEVICE void      pop_back()       { if (!empty()) --m_size; }

    OPG_INLINE OPG_HOSTDEVICE T*        data()           { return m_data; }
    OPG_INLINE OPG_HOSTDEVICE const T*  data()     const { return m_data; }

    OPG_INLINE OPG_HOSTDEVICE void      clear()          { m_size = 0; }

    // Random access
    OPG_INLINE OPG_HOSTDEVICE T&        operator [] (uint32_t i)       { return m_data[i]; }
    OPG_INLINE OPG_HOSTDEVICE const T&  operator [] (uint32_t i) const { return m_data[i]; }

private:
    T           m_data[Capacity];
    uint32_t    m_size = 0;
};

} // end namespace opg
