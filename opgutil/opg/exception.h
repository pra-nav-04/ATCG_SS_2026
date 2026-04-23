#pragma once

#ifndef __CUDACC__
#include <glad/glad.h>
#endif // __CUDACC__

#include <stdexcept>
#include <sstream>


//------------------------------------------------------------------------------
//
// GL error-checking
//
//------------------------------------------------------------------------------

#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
        GLenum err = glGetError();                                             \
        if( err != GL_NO_ERROR )                                               \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "GL error '"                                                 \
               << opg::getGLErrorString(err)                                   \
               << "' at " << __FILE__ << ":" << __LINE__ << ": " << #call      \
               << std::endl                                                    \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    }                                                                          \
    while (0)


#define GL_CHECK_ERRORS( )                                                     \
    do                                                                         \
    {                                                                          \
        GLenum err = glGetError();                                             \
        if( err != GL_NO_ERROR )                                               \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "GL error '"                                                 \
               <<  opg::getGLErrorString(err)                                  \
               << "' at " << __FILE__ << ":" << __LINE__                       \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    }                                                                          \
    while (0)


//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix error '"                                              \
               << optixGetErrorName(res)                                       \
               << "' at " << __FILE__ << ":" << __LINE__ << ": " << #call      \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix error '"                                              \
               << optixGetErrorName(res)                                       \
               << "' at " << __FILE__ << ":" << __LINE__ << ": " << #call      \
               << std::endl << "Log:" << std::endl << log                      \
               << ( sizeof_log > sizeof( log ) ? "<TRUNCATED>" : "" )          \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error '"                                               \
               << cudaGetErrorString(error)                                    \
               << "' at " << __FILE__ << ":" << __LINE__ << ": " << #call      \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString(error)                                    \
               << "' at " << __FILE__ << ":" << __LINE__                       \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define OPG_CHECK( call )                                                      \
    do                                                                         \
    {                                                                          \
        auto result = call;                                                    \
        if (!result)                                                           \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Error at " << __FILE__ << ":" << __LINE__ << ": " << #call  \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )

#define OPG_CHECK_MSG( call, msg )                                             \
    do                                                                         \
    {                                                                          \
        auto result = call;                                                    \
        if (!result)                                                           \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << msg << std::endl                                             \
               << "Error at " << __FILE__ << ":" << __LINE__ << ": " << #call  \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )

#define OPG_ASSERT( cond )                                                     \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << __FILE__ << ":" << __LINE__ << ": " << #cond                 \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


#define OPG_ASSERT_MSG( cond, msg )                                            \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << msg << std::endl                                             \
               << "Condition failed at " << __FILE__ << ":" << __LINE__ << ": " << #cond \
               ;                                                               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


namespace opg {

#ifndef __CUDACC__

inline const char* getGLErrorString(GLenum error)
{
    switch(error)
    {
        case GL_NO_ERROR:            return "No error";
        case GL_INVALID_ENUM:        return "Invalid enum";
        case GL_INVALID_VALUE:       return "Invalid value";
        case GL_INVALID_OPERATION:   return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:       return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:                     return "Unknown GL error";
    }
}

inline void checkGLError()
{
    GLenum err = glGetError();
    if(err != GL_NO_ERROR)
    {
        std::ostringstream oss;
        do
        {
            oss << "GL error: " << getGLErrorString(err) << std::endl;
            err = glGetError();
        }
        while(err != GL_NO_ERROR);

        throw std::runtime_error(oss.str());
    }
}

#endif // __CUDACC__

} // end namespace opg
