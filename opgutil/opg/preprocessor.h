#pragma once

#if defined(__CUDACC__)
#    define OPG_HOSTDEVICE __host__ __device__
#    define OPG_INLINE __forceinline__
#    define OPG_MEMBER_INIT( ... )
#else
#    define OPG_HOSTDEVICE
#    define OPG_INLINE inline
#    define OPG_MEMBER_INIT( ... ) = __VA_ARGS__
#endif

#ifndef M_PIf
#define M_PIf ((float)M_PI)
#endif
