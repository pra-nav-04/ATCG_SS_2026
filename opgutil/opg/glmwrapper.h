#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
// Nvidia compiler is bitching about defaulted functions, so just use explicit function bodies...
//#define GLM_FORCE_CTOR_INIT
// Well, if we want to use glm datatypes in __constant__ memory, their constructors must not initialize their contents to zero by default.
// `GLM_FORCE_NO_DEFAULTED_FUNCTIONS` is not a standard glm flag, but was added in our copy of glm...
#define GLM_FORCE_NO_DEFAULTED_FUNCTIONS
#endif // __CUDACC__

// Makes debug information from GDB easier to read
#define GLM_FORCE_XYZW_ONLY

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_cross_product.hpp>
#include <glm/gtx/exterior_product.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/vec_swizzle.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/norm.hpp>
#ifndef __CUDACC__
#include <glm/gtx/io.hpp>
#endif // __CUDACC__

namespace glm {

    namespace detail {

    template<typename V, typename T, bool Aligned>
    struct compute_sum{};

    template<typename T, qualifier Q, bool Aligned>
    struct compute_sum<vec<1, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<1, T, Q> const& a)
        {
            return a.x;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_sum<vec<2, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<2, T, Q> const& a)
        {
            return a.x + a.y;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_sum<vec<3, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<3, T, Q> const& a)
        {
            return a.x + a.y + a.z;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_sum<vec<4, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<4, T, Q> const& a)
        {
            return (a.x + a.y) + (a.z + a.w);
        }
    };

    template<typename V, typename T, bool Aligned>
    struct compute_prod{};

    template<typename T, qualifier Q, bool Aligned>
    struct compute_prod<vec<1, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<1, T, Q> const& a)
        {
            return a.x;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_prod<vec<2, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<2, T, Q> const& a)
        {
            return a.x * a.y;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_prod<vec<3, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<3, T, Q> const& a)
        {
            return a.x * a.y * a.z;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_prod<vec<4, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<4, T, Q> const& a)
        {
            return (a.x * a.y) * (a.z * a.w);
        }
    };



    template<typename V, typename T, bool Aligned>
    struct compute_minimum{};

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum<vec<1, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<1, T, Q> const& a)
        {
            return a.x;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum<vec<2, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<2, T, Q> const& a)
        {
            return min(a.x, a.y);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum<vec<3, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<3, T, Q> const& a)
        {
            return min(min(a.x, a.y), a.z);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum<vec<4, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<4, T, Q> const& a)
        {
            return min(min(a.x, a.y), min(a.z, a.w));
        }
    };

    template<typename V, typename T, bool Aligned>
    struct compute_maximum{};

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum<vec<1, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<1, T, Q> const& a)
        {
            return a.x;
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum<vec<2, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<2, T, Q> const& a)
        {
            return max(a.x, a.y);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum<vec<3, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<3, T, Q> const& a)
        {
            return max(max(a.x, a.y), a.z);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum<vec<4, T, Q>, T, Aligned>
    {
        GLM_FUNC_QUALIFIER static T call(vec<4, T, Q> const& a)
        {
            return max(max(a.x, a.y), max(a.z, a.w));
        }
    };

    ///

    template<typename V, typename T, bool Aligned>
    struct compute_minimum_index{};

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum_index<vec<1, T, Q>, typename vec<1, T, Q>::length_type, Aligned>
    {
        typedef typename vec<1, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<1, T, Q> const& a)
        {
            return length_type(0);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum_index<vec<2, T, Q>, typename vec<2, T, Q>::length_type, Aligned>
    {
        typedef typename vec<2, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<2, T, Q> const& a)
        {
            return a.x < a.y ? length_type(0) : length_type(1);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum_index<vec<3, T, Q>, typename vec<3, T, Q>::length_type, Aligned>
    {
        typedef typename vec<3, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<3, T, Q> const& a)
        {
            if (a.x < a.y && a.x < a.z)
                return length_type(0);
            else if (a.y < a.z)
                return length_type(1);
            else
                return length_type(2);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_minimum_index<vec<4, T, Q>, typename vec<4, T, Q>::length_type, Aligned>
    {
        typedef typename vec<4, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<4, T, Q> const& a)
        {
            length_type i_xy = a.x < a.y ? length_type(0) : length_type(1);
            T v_xy = a.x < a.y ? a.x : a.y;
            length_type i_zw = a.z < a.w ? length_type(2) : length_type(3);
            T v_zw = a.z < a.w ? a.z : a.w;
            return v_xy < v_zw ? i_xy : i_zw;
        }
    };

    ///

    template<typename V, typename T, bool Aligned>
    struct compute_maximum_index{};

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum_index<vec<1, T, Q>, typename vec<1, T, Q>::length_type, Aligned>
    {
        typedef typename vec<1, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<1, T, Q> const& a)
        {
            return length_type(0);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum_index<vec<2, T, Q>, typename vec<2, T, Q>::length_type, Aligned>
    {
        typedef typename vec<2, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<2, T, Q> const& a)
        {
            return a.x > a.y ? length_type(0) : length_type(1);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum_index<vec<3, T, Q>, typename vec<3, T, Q>::length_type, Aligned>
    {
        typedef typename vec<3, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<3, T, Q> const& a)
        {
            if (a.x > a.y && a.x > a.z)
                return length_type(0);
            else if (a.y > a.z)
                return length_type(1);
            else
                return length_type(2);
        }
    };

    template<typename T, qualifier Q, bool Aligned>
    struct compute_maximum_index<vec<4, T, Q>, typename vec<4, T, Q>::length_type, Aligned>
    {
        typedef typename vec<4, T, Q>::length_type length_type;
        GLM_FUNC_QUALIFIER static length_type call(vec<4, T, Q> const& a)
        {
            length_type i_xy = a.x > a.y ? length_type(0) : length_type(1);
            T v_xy = a.x > a.y ? a.x : a.y;
            length_type i_zw = a.z > a.w ? length_type(2) : length_type(3);
            T v_zw = a.z > a.w ? a.z : a.w;
            return v_xy > v_zw ? i_xy : i_zw;
        }
    };

    } // namespace detail

    // Compute the sum over all components in a vector
    template<length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER T sum(vec<L, T, Q> const& x)
    {
        return detail::compute_sum<vec<L, T, Q>, T, detail::is_aligned<Q>::value>::call(x);
    }


    // Compute the product over all components in a vector
    template<length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER T prod(vec<L, T, Q> const& x)
    {
        return detail::compute_prod<vec<L, T, Q>, T, detail::is_aligned<Q>::value>::call(x);
    }

    // Compute the minimum over all components in a vector
    template<length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER T minimum(vec<L, T, Q> const& x)
    {
        return detail::compute_minimum<vec<L, T, Q>, T, detail::is_aligned<Q>::value>::call(x);
    }

    // Compute the maximum over all components in a vector
    template<length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER T maximum(vec<L, T, Q> const& x)
    {
        return detail::compute_maximum<vec<L, T, Q>, T, detail::is_aligned<Q>::value>::call(x);
    }

    // Compute the index of the minimum component in a vector
    template<length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER typename vec<L, T, Q>::length_type minimum_index(vec<L, T, Q> const& x)
    {
        return detail::compute_minimum_index<vec<L, T, Q>, typename vec<L, T, Q>::length_type, detail::is_aligned<Q>::value>::call(x);
    }

    // Compute the index of the maximum component in a vector
    template<length_t L, typename T, qualifier Q>
    GLM_FUNC_QUALIFIER typename vec<L, T, Q>::length_type maximum_index(vec<L, T, Q> const& x)
    {
        return detail::compute_maximum_index<vec<L, T, Q>, typename vec<L, T, Q>::length_type, detail::is_aligned<Q>::value>::call(x);
    }

} // namespace glm

//
// Conversion between glm and cuda vector types
//

#include "opg/preprocessor.h"

#pragma region cuda_type

template <int Dim, typename Scalar>
struct cuda_type
{
    static_assert(sizeof(Scalar) == 0, "Not a cuda type (i.e. int3 or float4)!");
};

template<> struct cuda_type<1, glm::i8> { typedef char1 vector_type; typedef char scalar_type; };
template<> struct cuda_type<2, glm::i8> { typedef char2 vector_type; typedef char scalar_type; };
template<> struct cuda_type<3, glm::i8> { typedef char3 vector_type; typedef char scalar_type; };
template<> struct cuda_type<4, glm::i8> { typedef char4 vector_type; typedef char scalar_type; };
template<> struct cuda_type<1, glm::u8> { typedef uchar1 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<2, glm::u8> { typedef uchar2 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<3, glm::u8> { typedef uchar3 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<4, glm::u8> { typedef uchar4 vector_type; typedef unsigned char scalar_type; };
template<> struct cuda_type<1, glm::i16> { typedef short1 vector_type; typedef short scalar_type; };
template<> struct cuda_type<2, glm::i16> { typedef short2 vector_type; typedef short scalar_type; };
template<> struct cuda_type<3, glm::i16> { typedef short3 vector_type; typedef short scalar_type; };
template<> struct cuda_type<4, glm::i16> { typedef short4 vector_type; typedef short scalar_type; };
template<> struct cuda_type<1, glm::u16> { typedef ushort1 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<2, glm::u16> { typedef ushort2 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<3, glm::u16> { typedef ushort3 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<4, glm::u16> { typedef ushort4 vector_type; typedef unsigned short scalar_type; };
template<> struct cuda_type<1, glm::i32> { typedef int1 vector_type; typedef int scalar_type; };
template<> struct cuda_type<2, glm::i32> { typedef int2 vector_type; typedef int scalar_type; };
template<> struct cuda_type<3, glm::i32> { typedef int3 vector_type; typedef int scalar_type; };
template<> struct cuda_type<4, glm::i32> { typedef int4 vector_type; typedef int scalar_type; };
template<> struct cuda_type<1, glm::u32> { typedef uint1 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<2, glm::u32> { typedef uint2 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<3, glm::u32> { typedef uint3 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<4, glm::u32> { typedef uint4 vector_type; typedef unsigned int scalar_type; };
template<> struct cuda_type<1, glm::i64> { typedef longlong1 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<2, glm::i64> { typedef longlong2 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<3, glm::i64> { typedef longlong3 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<4, glm::i64> { typedef longlong4 vector_type; typedef long long scalar_type; };
template<> struct cuda_type<1, glm::u64> { typedef ulonglong1 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<2, glm::u64> { typedef ulonglong2 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<3, glm::u64> { typedef ulonglong3 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<4, glm::u64> { typedef ulonglong4 vector_type; typedef unsigned long long scalar_type; };
template<> struct cuda_type<1, glm::f32> { typedef float1 vector_type; typedef float scalar_type; };
template<> struct cuda_type<2, glm::f32> { typedef float2 vector_type; typedef float scalar_type; };
template<> struct cuda_type<3, glm::f32> { typedef float3 vector_type; typedef float scalar_type; };
template<> struct cuda_type<4, glm::f32> { typedef float4 vector_type; typedef float scalar_type; };
template<> struct cuda_type<1, glm::f64> { typedef double1 vector_type; typedef double scalar_type; };
template<> struct cuda_type<2, glm::f64> { typedef double2 vector_type; typedef double scalar_type; };
template<> struct cuda_type<3, glm::f64> { typedef double3 vector_type; typedef double scalar_type; };
template<> struct cuda_type<4, glm::f64> { typedef double4 vector_type; typedef double scalar_type; };

#pragma endregion cuda_type

#pragma region make_cuda_type

template<int N, typename T>
struct make_cuda_type
{
    static_assert(sizeof(T) == 0, "Not a cuda type (i.e. int3 or float4)!");
};

template <> struct make_cuda_type<1, glm::i8> : public cuda_type<1, glm::i8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_char1(x); } };
template <> struct make_cuda_type<2, glm::i8> : public cuda_type<2, glm::i8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_char2(x, y); } };
template <> struct make_cuda_type<3, glm::i8> : public cuda_type<3, glm::i8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_char3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i8> : public cuda_type<4, glm::i8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_char4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u8> : public cuda_type<1, glm::u8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_uchar1(x); } };
template <> struct make_cuda_type<2, glm::u8> : public cuda_type<2, glm::u8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_uchar2(x, y); } };
template <> struct make_cuda_type<3, glm::u8> : public cuda_type<3, glm::u8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_uchar3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u8> : public cuda_type<4, glm::u8> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_uchar4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::i16> : public cuda_type<1, glm::i16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_short1(x); } };
template <> struct make_cuda_type<2, glm::i16> : public cuda_type<2, glm::i16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_short2(x, y); } };
template <> struct make_cuda_type<3, glm::i16> : public cuda_type<3, glm::i16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_short3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i16> : public cuda_type<4, glm::i16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_short4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u16> : public cuda_type<1, glm::u16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_ushort1(x); } };
template <> struct make_cuda_type<2, glm::u16> : public cuda_type<2, glm::u16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_ushort2(x, y); } };
template <> struct make_cuda_type<3, glm::u16> : public cuda_type<3, glm::u16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_ushort3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u16> : public cuda_type<4, glm::u16> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_ushort4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::i32> : public cuda_type<1, glm::i32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_int1(x); } };
template <> struct make_cuda_type<2, glm::i32> : public cuda_type<2, glm::i32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_int2(x, y); } };
template <> struct make_cuda_type<3, glm::i32> : public cuda_type<3, glm::i32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_int3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i32> : public cuda_type<4, glm::i32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_int4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u32> : public cuda_type<1, glm::u32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_uint1(x); } };
template <> struct make_cuda_type<2, glm::u32> : public cuda_type<2, glm::u32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_uint2(x, y); } };
template <> struct make_cuda_type<3, glm::u32> : public cuda_type<3, glm::u32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_uint3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u32> : public cuda_type<4, glm::u32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_uint4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::i64> : public cuda_type<1, glm::i64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_longlong1(x); } };
template <> struct make_cuda_type<2, glm::i64> : public cuda_type<2, glm::i64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_longlong2(x, y); } };
template <> struct make_cuda_type<3, glm::i64> : public cuda_type<3, glm::i64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_longlong3(x, y, z); } };
template <> struct make_cuda_type<4, glm::i64> : public cuda_type<4, glm::i64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_longlong4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::u64> : public cuda_type<1, glm::u64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_ulonglong1(x); } };
template <> struct make_cuda_type<2, glm::u64> : public cuda_type<2, glm::u64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_ulonglong2(x, y); } };
template <> struct make_cuda_type<3, glm::u64> : public cuda_type<3, glm::u64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_ulonglong3(x, y, z); } };
template <> struct make_cuda_type<4, glm::u64> : public cuda_type<4, glm::u64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_ulonglong4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::f32> : public cuda_type<1, glm::f32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_float1(x); } };
template <> struct make_cuda_type<2, glm::f32> : public cuda_type<2, glm::f32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_float2(x, y); } };
template <> struct make_cuda_type<3, glm::f32> : public cuda_type<3, glm::f32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_float3(x, y, z); } };
template <> struct make_cuda_type<4, glm::f32> : public cuda_type<4, glm::f32> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_float4(x, y, z, w); } };
template <> struct make_cuda_type<1, glm::f64> : public cuda_type<1, glm::f64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x) { return make_double1(x); } };
template <> struct make_cuda_type<2, glm::f64> : public cuda_type<2, glm::f64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_double2(x, y); } };
template <> struct make_cuda_type<3, glm::f64> : public cuda_type<3, glm::f64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z) { return make_double3(x, y, z); } };
template <> struct make_cuda_type<4, glm::f64> : public cuda_type<4, glm::f64> { static OPG_HOSTDEVICE OPG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w) { return make_double4(x, y, z, w); } };

#pragma endregion make_cuda_type

#pragma region glm2cuda

template <glm::length_t N, typename T, glm::qualifier Q>
struct glm2cuda_detail
{
    static_assert(sizeof(T) == 0, "Not implemented for this type!");
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<1, T, Q>
{
    static constexpr int N = 1;
    static OPG_HOSTDEVICE OPG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q> &v) { return make_cuda_type<N, T>::apply(v.x); };
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<2, T, Q>
{
    static constexpr int N = 2;
    static OPG_HOSTDEVICE OPG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q> &v) { return make_cuda_type<N, T>::apply(v.x, v.y); };
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<3, T, Q>
{
    static constexpr int N = 3;
    static OPG_HOSTDEVICE OPG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q> &v) { return make_cuda_type<N, T>::apply(v.x, v.y, v.z); };
};

template <typename T, glm::qualifier Q>
struct glm2cuda_detail<4, T, Q>
{
    static constexpr int N = 4;
    static OPG_HOSTDEVICE OPG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q> &v) { return make_cuda_type<N, T>::apply(v.x, v.y, v.z, v.w); };
};

template <glm::length_t N, typename T, glm::qualifier Q>
OPG_HOSTDEVICE OPG_INLINE typename cuda_type<N, T>::vector_type glm2cuda(const glm::vec<N, T, Q> &v)
{
    return glm2cuda_detail<N, T, Q>::apply(v);
}

#pragma endregion glm2cuda

#pragma region inv_cuda_type

template <typename VecType>
struct inv_cuda_type
{
    static_assert(sizeof(VecType) == 0, "Not a cuda type!");
};

template<> struct inv_cuda_type<char1> { static constexpr int dim = 1; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<char2> { static constexpr int dim = 2; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<char3> { static constexpr int dim = 3; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<char4> { static constexpr int dim = 4; typedef glm::i8 scalar_type; };
template<> struct inv_cuda_type<uchar1> { static constexpr int dim = 1; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<uchar2> { static constexpr int dim = 2; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<uchar3> { static constexpr int dim = 3; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<uchar4> { static constexpr int dim = 4; typedef glm::u8 scalar_type; };
template<> struct inv_cuda_type<short1> { static constexpr int dim = 1; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<short2> { static constexpr int dim = 2; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<short3> { static constexpr int dim = 3; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<short4> { static constexpr int dim = 4; typedef glm::i16 scalar_type; };
template<> struct inv_cuda_type<ushort1> { static constexpr int dim = 1; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<ushort2> { static constexpr int dim = 2; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<ushort3> { static constexpr int dim = 3; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<ushort4> { static constexpr int dim = 4; typedef glm::u16 scalar_type; };
template<> struct inv_cuda_type<int1> { static constexpr int dim = 1; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<int2> { static constexpr int dim = 2; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<int3> { static constexpr int dim = 3; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<int4> { static constexpr int dim = 4; typedef glm::i32 scalar_type; };
template<> struct inv_cuda_type<uint1> { static constexpr int dim = 1; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<uint2> { static constexpr int dim = 2; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<uint3> { static constexpr int dim = 3; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<uint4> { static constexpr int dim = 4; typedef glm::u32 scalar_type; };
template<> struct inv_cuda_type<longlong1> { static constexpr int dim = 1; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<longlong2> { static constexpr int dim = 2; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<longlong3> { static constexpr int dim = 3; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<longlong4> { static constexpr int dim = 4; typedef glm::i64 scalar_type; };
template<> struct inv_cuda_type<ulonglong1> { static constexpr int dim = 1; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<ulonglong2> { static constexpr int dim = 2; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<ulonglong3> { static constexpr int dim = 3; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<ulonglong4> { static constexpr int dim = 4; typedef glm::u64 scalar_type; };
template<> struct inv_cuda_type<float1> { static constexpr int dim = 1; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<float2> { static constexpr int dim = 2; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<float3> { static constexpr int dim = 3; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<float4> { static constexpr int dim = 4; typedef glm::f32 scalar_type; };
template<> struct inv_cuda_type<double1> { static constexpr int dim = 1; typedef glm::f64 scalar_type; };
template<> struct inv_cuda_type<double2> { static constexpr int dim = 2; typedef glm::f64 scalar_type; };
template<> struct inv_cuda_type<double3> { static constexpr int dim = 3; typedef glm::f64 scalar_type; };
template<> struct inv_cuda_type<double4> { static constexpr int dim = 4; typedef glm::f64 scalar_type; };

template<> struct inv_cuda_type<dim3> { static constexpr int dim = 3; typedef glm::u32 scalar_type; };

#pragma endregion inv_cuda_type

#pragma region cuda2glm

template <int N, typename T, glm::qualifier Q>
struct cuda2glm_detail
{
    static_assert(sizeof(T) == 0, "Not implemented for type!");
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<1, T, Q>
{
    typedef glm::vec<1, T, Q> result_type;
    static OPG_HOSTDEVICE OPG_INLINE result_type apply(const typename cuda_type<1, T>::vector_type &v) { return result_type(v.x); }
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<2, T, Q>
{
    typedef glm::vec<2, T, Q> result_type;
    static OPG_HOSTDEVICE OPG_INLINE result_type apply(const typename cuda_type<2, T>::vector_type &v) { return result_type(v.x, v.y); }
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<3, T, Q>
{
    typedef glm::vec<3, T, Q> result_type;
    static OPG_HOSTDEVICE OPG_INLINE result_type apply(const typename cuda_type<3, T>::vector_type &v) { return result_type(v.x, v.y, v.z); }
};

template <typename T, glm::qualifier Q>
struct cuda2glm_detail<4, T, Q>
{
    typedef glm::vec<4, T, Q> result_type;
    static OPG_HOSTDEVICE OPG_INLINE result_type apply(const typename cuda_type<4, T>::vector_type &v) { return result_type(v.x, v.y, v.z, v.w); }
};

template <typename CudaVectorType, glm::qualifier Q = glm::defaultp>
OPG_HOSTDEVICE OPG_INLINE typename cuda2glm_detail<inv_cuda_type<CudaVectorType>::dim, typename inv_cuda_type<CudaVectorType>::scalar_type, Q>::result_type cuda2glm(const CudaVectorType &v)
{
    return cuda2glm_detail<inv_cuda_type<CudaVectorType>::dim, typename inv_cuda_type<CudaVectorType>::scalar_type, Q>::apply(v);
};

#ifdef __CUDACC__

template <typename T>
__device__ OPG_INLINE T tex1D(cudaTextureObject_t tex, glm::vec1 coord)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return tex1D<float>(tex, coord.x);
    }
    else if constexpr (T::length() == 3)
    {
        return glm::xyz(cuda2glm(tex1D<float4>(tex, coord.x)));
    }
    else
    {
        return cuda2glm(tex1D<decltype(glm2cuda(T()))>(tex, coord.x));
    }
}

template <typename T>
__device__ OPG_INLINE T tex2D(cudaTextureObject_t tex, glm::vec2 coord)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return tex2D<float>(tex, coord.x, coord.y);
    }
    else if constexpr (T::length() == 3)
    {
        return glm::xyz(cuda2glm(tex2D<float4>(tex, coord.x, coord.y)));
    }
    else
    {
        return cuda2glm(tex2D<decltype(glm2cuda(T()))>(tex, coord.x, coord.y));
    }
}

template <typename T>
__device__ OPG_INLINE T tex3D(cudaTextureObject_t tex, glm::vec3 coord)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return tex3D<float>(tex, coord.x, coord.y, coord.z);
    }
    else if constexpr (T::length() == 3)
    {
        return glm::xyz(cuda2glm(tex3D<float4>(tex, coord.x, coord.y, coord.z)));
    }
    else
    {
        return cuda2glm(tex3D<decltype(glm2cuda(T()))>(tex, coord.x, coord.y, coord.z));
    }
}

#endif // __CUDACC__

namespace std {

/// Numeric limits
template <int N, typename T, glm::qualifier Q>
struct numeric_limits<glm::vec<N, T, Q>>
{
    static bool const is_specialized = numeric_limits<T>::is_specialized;
    static bool const is_signed = numeric_limits<T>::is_signed;
    static bool const is_integer = numeric_limits<T>::is_integer;
    static bool const is_exact = numeric_limits<T>::is_exact;
    static bool const has_infinity = numeric_limits<T>::has_infinity;
    static bool const has_quiet_NaN = numeric_limits<T>::has_quiet_NaN;
    static bool const has_signaling_NaN = numeric_limits<T>::has_signaling_NaN;
    //static std::float_denorm_style const has_denorm = numeric_limits<T>::has_denorm;
    static bool const has_denorm_loss = numeric_limits<T>::has_denorm_loss;
    // static std::float_round_style const round_style = std::round_to_nearest;
    static bool const is_iec559 = numeric_limits<T>::is_iec559;
    static bool const is_bounded = numeric_limits<T>::is_bounded;
    static bool const is_modulo = numeric_limits<T>::is_modulo;
    static int const digits = numeric_limits<T>::digits;

    typedef glm::vec<N, T, Q> vec_type;

    /// Least positive value
    OPG_HOSTDEVICE static constexpr vec_type min()
    {
        T nested = numeric_limits<T>::min();
        vec_type result = vec_type{nested};
        return result;
    }

    /// Minimum finite value
    OPG_HOSTDEVICE static vec_type lowest() { return vec_type(numeric_limits<T>::lowest()); }

    /// Maximum finite value
    OPG_HOSTDEVICE
    static vec_type max() { return vec_type(numeric_limits<T>::max()); }

    /// Returns smallest finite value
    OPG_HOSTDEVICE
    static vec_type epsilon() { return vec_type(numeric_limits<T>::epsilon()); }

    /// Returns smallest finite value
    OPG_HOSTDEVICE
    static vec_type round_error() { return vec_type(numeric_limits<T>::round_error()); }

    /// Returns smallest finite value
    OPG_HOSTDEVICE
    static vec_type infinity() { return vec_type(numeric_limits<T>::infinity()); }

    /// Returns smallest finite value
    OPG_HOSTDEVICE
    static vec_type quiet_NaN() { return vec_type(numeric_limits<T>::quiet_NaN()); }

    /// Returns smallest finite value
    OPG_HOSTDEVICE
    static vec_type signaling_NaN() { return vec_type(numeric_limits<T>::signaling_NaN()); }

    /// Returns smallest finite value
    OPG_HOSTDEVICE static vec_type denorm_min() { return vec_type(numeric_limits<T>::denorm_min()); }
};

} // namespace std

#pragma endregion cuda2glm
