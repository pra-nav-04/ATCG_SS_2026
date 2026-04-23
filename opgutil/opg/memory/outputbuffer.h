#pragma once

#include <glad/glad.h> // Needs to be included before gl_interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <optix.h>

#include <iostream>
#include <vector>

#include "opg/exception.h"
#include "opg/preprocessor.h"

namespace opg {

enum class OutputBufferType
{
    CUDA_DEVICE = 0, // not preferred, typically slower than `HOST`, data is explicitly copied from device to host buffer
    GL_INTEROP  = 1, // single device only, preferred for single device. NOTE: make sure that you are not using the igpu on mobile devices!
    CUDA_HOST   = 2  // cuda memory is stored in host memory and is therefore direcly accessible by host cpu (no copying)
};


template <typename T>
class OutputBuffer
{
public:
    OutputBuffer(OutputBufferType type, uint32_t width, uint32_t height);
    ~OutputBuffer();

    OPG_INLINE void setStream(CUstream stream) { m_stream = stream; }

    void resize(uint32_t width, uint32_t height);

    // Allocate or update device pointer as necessary for CUDA access
    T* mapCUDA();
    void unmapCUDA();

    OPG_INLINE uint32_t width()  const { return m_width;  }
    OPG_INLINE uint32_t height() const { return m_height; }

    // Get output buffer
    GLuint getPBO();
    T* getHostPointer();

private:
    OutputBufferType           m_type;

    uint32_t                   m_width             = 0u;
    uint32_t                   m_height            = 0u;

    cudaGraphicsResource*      m_cuda_gfx_resource = nullptr;
    GLuint                     m_pbo               = 0u;
    T*                         m_device_pixels     = nullptr;
    T*                         m_host_pixels       = nullptr;
    std::vector<T>             m_host_pixels_storage;

    CUstream                   m_stream            = 0u;
    int32_t                    m_device_idx        = 0;
};


template <typename T>
OutputBuffer<T>::OutputBuffer(OutputBufferType type, uint32_t width, uint32_t height) :
    m_type { type }
{
    // If using GL Interop, expect that the active device is also the display device.
    if (type == OutputBufferType::GL_INTEROP)
    {
        int current_device, is_display_device;
        CUDA_CHECK( cudaGetDevice(&current_device) );
        CUDA_CHECK( cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device) );
        if (!is_display_device)
        {
            throw std::runtime_error(
                    "GL interop is only available on display device, please use display device for optimal "
                    "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
                    "degraded performance."
                    );
        }
    }
    resize(width, height);
}


template <typename T>
OutputBuffer<T>::~OutputBuffer()
{
    try
    {
        switch (m_type)
        {
            case OutputBufferType::CUDA_DEVICE:
            {
                CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_device_pixels)) );
            }
            break;
            case OutputBufferType::GL_INTEROP:
            {
                CUDA_CHECK( cudaGraphicsUnregisterResource(m_cuda_gfx_resource) );
            }
            break;
            case OutputBufferType::CUDA_HOST:
            {
                CUDA_CHECK( cudaFreeHost(reinterpret_cast<void*>(m_host_pixels) ) );
            }
            break;
        }

        if (m_pbo != 0u)
        {
            GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, 0) );
            GL_CHECK( glDeleteBuffers(1, &m_pbo) );
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "CUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
    }
}


template <typename T>
void OutputBuffer<T>::resize(uint32_t width, uint32_t height)
{
    if( m_width == width && m_height == height )
        return;

    m_width  = width;
    m_height = height;

    switch (m_type)
    {
        case OutputBufferType::CUDA_DEVICE:
        {
            CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_device_pixels)) );
            CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>(&m_device_pixels),
                sizeof(T)*m_width*m_height
            ) );
        }
        break;
        case OutputBufferType::CUDA_HOST:
        {
            CUDA_CHECK( cudaFreeHost(reinterpret_cast<void*>(m_host_pixels)) );
            CUDA_CHECK( cudaHostAlloc(
                reinterpret_cast<void**>(&m_host_pixels),
                sizeof(T)*m_width*m_height,
                cudaHostAllocPortable | cudaHostAllocMapped
            ) );
            CUDA_CHECK( cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&m_device_pixels),
                reinterpret_cast<void*>(m_host_pixels),
                0 /*flags*/
            ) );
        }
        break;
        case OutputBufferType::GL_INTEROP:
        {
            if (m_pbo == 0u)
            {
                GL_CHECK( glGenBuffers(1, &m_pbo) );
            }

            // GL buffer gets resized below
            GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_pbo) );
            GL_CHECK( glBufferData(GL_ARRAY_BUFFER, sizeof(T)*m_width*m_height, nullptr, GL_STREAM_DRAW) );
            GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, 0u) );

            CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
                &m_cuda_gfx_resource,
                m_pbo,
                cudaGraphicsMapFlagsWriteDiscard
            ) );
        }
        break;
    }

    if (m_type != OutputBufferType::GL_INTEROP && m_pbo != 0u)
    {
        GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_pbo) );
        GL_CHECK( glBufferData(GL_ARRAY_BUFFER, sizeof(T)*m_width*m_height, nullptr, GL_STREAM_DRAW) );
        GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, 0u) );
    }

    if (!m_host_pixels_storage.empty())
        m_host_pixels_storage.resize(m_width*m_height);
}


template <typename T>
T* OutputBuffer<T>::mapCUDA()
{
    switch (m_type)
    {
        case OutputBufferType::CUDA_DEVICE:
        case OutputBufferType::CUDA_HOST:
        {
            // nothing needed
        }
        break;
        case OutputBufferType::GL_INTEROP:
        {
            size_t buffer_size = 0u;
            CUDA_CHECK( cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream) );
            CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&m_device_pixels),
                &buffer_size,
                m_cuda_gfx_resource
            ) );
        }
        break;
    }

    return m_device_pixels;
}


template <typename T>
void OutputBuffer<T>::unmapCUDA()
{
    switch (m_type)
    {
        case OutputBufferType::CUDA_DEVICE:
        case OutputBufferType::CUDA_HOST:
        {
            CUDA_CHECK( cudaStreamSynchronize(m_stream) );
        }
        break;
        case OutputBufferType::GL_INTEROP:
        {
            CUDA_CHECK( cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource,  m_stream) );
        }
        break;
    }
}

template <typename T>
GLuint OutputBuffer<T>::getPBO()
{
    if (m_pbo == 0u)
        GL_CHECK( glGenBuffers(1, &m_pbo) );

    const size_t buffer_size = sizeof(T)*m_width*m_height;

    switch (m_type)
    {
        case OutputBufferType::CUDA_DEVICE:
        {
            // We need a host buffer to act as a way-station
            if (m_host_pixels_storage.empty())
                m_host_pixels_storage.resize(m_width*m_height);

            m_host_pixels = m_host_pixels_storage.data();

            CUDA_CHECK( cudaMemcpy(
                static_cast<void*>(m_host_pixels),
                m_device_pixels,
                buffer_size,
                cudaMemcpyDeviceToHost
            ) );

            GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
            GL_CHECK( glBufferData(
                GL_ARRAY_BUFFER,
                buffer_size,
                static_cast<void*>(m_host_pixels),
                GL_STREAM_DRAW
            ) );
            GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, 0) );
        }
        break;
        case OutputBufferType::CUDA_HOST:
        {
            GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_pbo) );
            GL_CHECK( glBufferData(
                GL_ARRAY_BUFFER,
                buffer_size,
                static_cast<void*>(m_host_pixels),
                GL_STREAM_DRAW
            ) );
            GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, 0) );
        }
        break;
        case OutputBufferType::GL_INTEROP:
        {
            // Nothing needed
        }
        break;
    }

    return m_pbo;
}


template <typename T>
T* OutputBuffer<T>::getHostPointer()
{
    switch (m_type)
    {
        case OutputBufferType::CUDA_DEVICE:
        case OutputBufferType::GL_INTEROP:
        {
            m_host_pixels_storage.resize( m_width*m_height );
            m_host_pixels = m_host_pixels_storage.data();

            mapCUDA(); // make sure that m_device_pixels contains valid pointer
            CUDA_CHECK( cudaMemcpy(
                static_cast<void*>(m_host_pixels),
                m_device_pixels,
                sizeof(T)*m_width*m_height,
                cudaMemcpyDeviceToHost
            ) );
            unmapCUDA();
        }
        break;
        case OutputBufferType::CUDA_HOST:
        {
            // Nothing needed
        }
        break;
    }

    return m_host_pixels;
}

} // end namespace opg
