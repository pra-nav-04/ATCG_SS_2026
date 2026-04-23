#pragma once

#include <stdint.h>

#include "opg/preprocessor.h"

namespace opg {

template <typename T>
struct BufferView
{
    std::byte*  data            OPG_MEMBER_INIT( nullptr );
    uint32_t    count           OPG_MEMBER_INIT( 0 );
    uint16_t    byte_stride     OPG_MEMBER_INIT( sizeof(T) );
    uint16_t    elmt_byte_size  OPG_MEMBER_INIT( sizeof(T) );

    BufferView() = default;
    OPG_INLINE OPG_HOSTDEVICE BufferView(std::byte* d, uint32_t c, uint16_t b=sizeof(T), uint16_t e=sizeof(T)) : data{d}, count{c}, byte_stride{b}, elmt_byte_size{e}
    {}

    OPG_INLINE OPG_HOSTDEVICE bool isValid() const
    {
        return static_cast<bool>( data );
    }

    OPG_INLINE OPG_HOSTDEVICE operator bool() const
    {
        return isValid();
    }

    // NOTE: data is non-const here!
    // Use BufferView<const T> instead.
    OPG_INLINE OPG_HOSTDEVICE T& operator[]( uint32_t idx ) const
    {
        return *reinterpret_cast<T*>( data + idx*(byte_stride ? byte_stride : sizeof( T ) ) );
    }

    OPG_INLINE OPG_HOSTDEVICE T& operator[]( uint32_t idx )
    {
        return *reinterpret_cast<T*>( data + idx*(byte_stride ? byte_stride : sizeof( T ) ) );
    }

    template <typename U>
    BufferView<U> reinterpret(uint32_t offset = 0) const
    {
        BufferView<U> result;
        result.data = data + offset;
        result.byte_stride = byte_stride;
        result.count = count;
        return result;
    }
};

// typedef BufferView<std::byte> GenericBufferView;
struct GenericBufferView : BufferView<std::byte>
{
    GenericBufferView() = default;
    GenericBufferView(const BufferView<std::byte> &other) :
        BufferView<std::byte>(other)
    {}

    template<typename T>
    OPG_INLINE OPG_HOSTDEVICE const BufferView<T> asType() const
    {
        BufferView<T> result;
        result.data = data;
        result.count = count;
        result.byte_stride = byte_stride;
        result.elmt_byte_size = elmt_byte_size;
        return result;
    }

    template<typename T>
    OPG_INLINE OPG_HOSTDEVICE BufferView<T> asType()
    {
        BufferView<T> result;
        result.data = data;
        result.count = count;
        result.byte_stride = byte_stride;
        result.elmt_byte_size = elmt_byte_size;
        return result;
    }
};

} // end namespace opg
