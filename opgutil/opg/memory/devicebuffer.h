#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "opg/exception.h"
#include "opg/memory/bufferview.h"

namespace opg {

template <typename T = std::byte>
class DeviceBuffer
{
public:
    DeviceBuffer(size_t count = 0) {
        alloc(count);
    }
    ~DeviceBuffer() {
        free();
    }

    template <typename U>
    friend class DeviceBuffer;

    template <typename U>
    DeviceBuffer(DeviceBuffer<U> && other) :
        m_size { (other.m_size * sizeof(U)) / sizeof(T) },
        m_capacity { (other.m_capacity * sizeof(U)) / sizeof(T) },
        m_ptr { reinterpret_cast<T*>(other.m_ptr) }
    {
        if ((other.m_size * sizeof(U)) % sizeof(T) != 0)
        {
            m_size      = 0;
            m_capacity  = 0;
            m_ptr       = nullptr;
            throw std::runtime_error("buffer size not compatible!");
        }
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_ptr      = nullptr;
    }

    DeviceBuffer(const DeviceBuffer<T> &) = delete;
    DeviceBuffer(DeviceBuffer<T> && other) :
        m_size { other.m_size },
        m_capacity { other.m_capacity },
        m_ptr { other.m_ptr }
    {
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_ptr      = nullptr;
    }

    DeviceBuffer<T> &operator = (const DeviceBuffer<T> &) = delete;
    DeviceBuffer<T> &operator = (DeviceBuffer<T> && other)
    {
        free();
        m_size           = other.m_size;
        m_capacity       = other.m_capacity;
        m_ptr            = other.m_ptr;
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_ptr      = nullptr;
        return *this;
    }

    void alloc(size_t count)
    {
        free();
        m_capacity = m_size = count;
        if (m_size)
        {
            CUDA_CHECK( cudaMalloc(&m_ptr, m_capacity * sizeof(T)) );
        }
    }
    void allocIfRequired(size_t count)
    {
        if (count <= m_size)
        {
            m_size = count;
            return;
        }
        alloc(count);
    }
    void free()
    {
        m_size      = 0;
        m_capacity  = 0;
        CUDA_CHECK( cudaFree(m_ptr) );
        m_ptr = nullptr;
    }
    inline void clear() { free(); }
    T *release()
    {
        T *current  = m_ptr;
        m_size      = 0;
        m_capacity  = 0;
        m_ptr       = nullptr;
        return current;
    }

    void upload(const T *data)
    {
        CUDA_CHECK( cudaMemcpy(m_ptr, data, m_size * sizeof(T), cudaMemcpyHostToDevice) );
    }
    void uploadSub(const T *data, size_t count, size_t offset = 0ull)
    {
        OPG_ASSERT( count + offset <= m_capacity );
        CUDA_CHECK( cudaMemcpy(m_ptr + offset, data, count * sizeof(T), cudaMemcpyHostToDevice) );
    }

    void download(T *data) const
    {
        CUDA_CHECK( cudaMemcpy(data, m_ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost) );
    }
    void downloadSub(T *data, size_t count, size_t offset = 0ull) const
    {
        OPG_ASSERT( count + offset <= m_capacity );
        CUDA_CHECK( cudaMemcpy(data, m_ptr + offset, count * sizeof(T), cudaMemcpyDeviceToHost) );
    }

    inline CUdeviceptr getRaw() const { return reinterpret_cast<CUdeviceptr>(m_ptr); }
    inline CUdeviceptr getRaw(size_t index) const { return reinterpret_cast<CUdeviceptr>(m_ptr + index); }

    inline T *data() const { return m_ptr; }
    inline T *data(size_t index) const { return m_ptr + index; }

    inline size_t size() const { return m_size; }
    inline size_t capacity() const { return m_capacity; }
    inline size_t byteSize() const { return m_capacity * sizeof(T); }

    inline BufferView<T> view()
    {
        BufferView<T> buffer_view;
        buffer_view.data = reinterpret_cast<std::byte*>(this->data());
        buffer_view.count = static_cast<uint32_t>(this->size());
        buffer_view.byte_stride = sizeof(T);
        buffer_view.elmt_byte_size = sizeof(T);
        return buffer_view;
    }

  private:
    size_t m_size     = 0;
    size_t m_capacity = 0;
    T*     m_ptr      = nullptr;
};

typedef DeviceBuffer<std::byte> GenericDeviceBuffer;

} // end namespace opg
