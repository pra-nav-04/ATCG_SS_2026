#pragma once

#include <cstdint>
#include "opg/preprocessor.h"

//
// Helper functions for packing and unpacking 64-bit pointers into two 32-bit integers.
//

OPG_INLINE OPG_HOSTDEVICE void* unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


OPG_INLINE OPG_HOSTDEVICE void packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}
