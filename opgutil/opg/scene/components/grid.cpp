#include "opg/scene/components/grid.h"
#include "opg/scene/sceneloader.h"
#include "opg/kernels/texture.h"

#include <npy.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ranges.h>


namespace opg{

static cudaTextureAddressMode decode_address_mode(const std::string &address_mode_name)
{
    if (address_mode_name == "wrap")
    {
        return cudaAddressModeWrap;
    }
    else if (address_mode_name == "clamp")
    {
        return cudaAddressModeClamp;
    }
    else if (address_mode_name == "mirror")
    {
        return cudaAddressModeMirror;
    }
    else if (address_mode_name == "border")
    {
        return cudaAddressModeBorder;
    }
    else
    {
        throw std::runtime_error("Invalid address mode " + address_mode_name);
    }
}

static cudaTextureFilterMode decode_filter_mode(const std::string &filter_mode_name)
{
    if (filter_mode_name == "linear")
    {
        return cudaFilterModeLinear;
    }
    else if (filter_mode_name == "nearest")
    {
        return cudaFilterModePoint;
    }
    else
    {
        throw std::runtime_error("Invalid filter mode " + filter_mode_name);
    }
}


GridComponent::GridComponent(PrivatePtr<Scene> _scene, const Properties &_props) :
    SceneComponent(std::move(_scene), _props)
{
    m_filter_mode = decode_filter_mode(_props.getString("filter_mode", "linear"));
    m_address_mode = decode_address_mode(_props.getString("address_mode", "wrap"));
    m_border_color = _props.getVector("border_color", glm::vec4(0, 0, 0, 1));
    m_srgb = _props.getBool("srgb", false);

    std::string filename = _props.getPath() + _props.getString("filename", "");
    if (!std::filesystem::exists(filename))
    {
        throw std::runtime_error("Input file does not exist");
    }
    if (!tryLoadVol(filename))
        if (!tryLoadNpy(filename))
            throw std::runtime_error("Failed to load texture...");

    computeMaximum();

    m_to_uvw = _props.getMatrix("to_uv", glm::identity<glm::mat4>());
}

GridComponent::~GridComponent()
{
    cudaDestroyTextureObject(m_texture);
    cudaFreeArray(m_array);
}


bool GridComponent::tryLoadNpy(const std::string &filename)
{
    npy::dtype_t numpy_dtype;
    std::vector<unsigned long> numpy_shape;
    bool numpy_fortran_order;
    std::vector<char> numpy_data;
    try
    {
        npy::LoadUnknownFromNumpy(filename, numpy_dtype, numpy_shape, numpy_fortran_order, numpy_data);
    }
    catch (...)
    {
        // Soft fail.
        return false;
    }

    if (numpy_fortran_order)
        throw std::runtime_error("Numpy fortran order is not supported.");

    if (numpy_shape.size() != 4)
        throw std::runtime_error(fmt::format("Numpy array has incompatible shape! Shape should be (d, h, w, c), got {} instead", numpy_shape));

    if (numpy_shape.back() > 4)
        throw std::runtime_error(fmt::format("Unsupported channel count! Shape should be (d, h, w, c), got {} instead", numpy_shape));

    if (npy::has_typestring<float>::dtype.tie() != numpy_dtype.tie())
        throw std::runtime_error("Only 32bit float arrays supported currenlty!");


    // TODO check which dimension is which!?
    m_channels = numpy_shape[3];
    m_extent = glm::uvec3(numpy_shape[2], numpy_shape[1], numpy_shape[0]);

    cudaChannelFormatDesc channelDesc;
    switch (m_channels)
    {
        case 1:
        {
            // 1 channel float
            channelDesc = cudaCreateChannelDesc<float1>();
            break;
        }
        case 2:
        {
            // 2 channel float
            channelDesc = cudaCreateChannelDesc<float2>();
            break;
        }
        case 3:
        {
            // 3 channel float
            // TODO 3 channel float is not supported by CUDA!?
            std::vector<char> padded_data(glm::prod(m_extent)*4*sizeof(float)); // four channels!
            float *numpy_data_float_ptr = reinterpret_cast<float*>(numpy_data.data());
            float *padded_data_float_ptr = reinterpret_cast<float*>(padded_data.data());
            for (uint32_t i = 0; i < glm::prod(m_extent); ++i)
            {
                for (uint32_t j = 0; j < 3; ++j)
                    padded_data_float_ptr[i*4 + j] = numpy_data_float_ptr[i*3 + j];
                padded_data_float_ptr[i*4+3] = 0.0f;
            }
            numpy_data = std::move(padded_data);
            m_channels = 4;
            channelDesc = cudaCreateChannelDesc<float4>();
            break;
        }
        case 4:
        {
            // 4 channel float
            channelDesc = cudaCreateChannelDesc<float4>();
            break;
        }
        default:
        {
            throw std::runtime_error("Unsupported channel count!");
        }
    }

    cudaExtent cudaExtent { m_extent.x, m_extent.y, m_extent.z };

    CUDA_CHECK( cudaMalloc3DArray(&m_array, &channelDesc, cudaExtent) );

    cudaMemcpy3DParms copyParams = {};
    // TODO datatype size dynamic here
    copyParams.srcPtr = make_cudaPitchedPtr(numpy_data.data(), m_extent.x*m_channels*numpy_dtype.itemsize, m_extent.x, m_extent.y);
    copyParams.dstArray = m_array;
    copyParams.extent = cudaExtent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK( cudaMemcpy3D(&copyParams) );

    cudaResourceDesc resDesc;
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_array;

    cudaTextureDesc texDesc;
    texDesc.addressMode[0] = m_address_mode;
    texDesc.addressMode[1] = m_address_mode;
    texDesc.addressMode[2] = m_address_mode;
    texDesc.filterMode = m_filter_mode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.sRGB = m_srgb ? 1 : 0;
    texDesc.borderColor[0] = m_border_color.x;
    texDesc.borderColor[1] = m_border_color.y;
    texDesc.borderColor[2] = m_border_color.z;
    texDesc.borderColor[3] = m_border_color.w;
    texDesc.normalizedCoords = true; // Always!
    texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.mipmapLevelBias = 0;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = 0;
    texDesc.disableTrilinearOptimization = 0;
#if CUDA_VERSION >= 12000 || (CUDA_VERSION >= 11060 && CUDA_VERSION < 11080)
    texDesc.seamlessCubemap = 0;
#endif

    CUDA_CHECK( cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr) );

    return true;
}

bool GridComponent::tryLoadVol(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    struct VolFileHeader
    {
        uint32_t magic_number;
        uint32_t data_type;
        uint32_t size_x;
        uint32_t size_y;
        uint32_t size_z;
        uint32_t channels;
        float bbox_corners[6];
    };

    // Read the header
    VolFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Check the magic number
    if (header.magic_number != 0x034C4F56) // "VOL\x03"
        return false;

    if (header.data_type != 1)
        throw std::runtime_error("Only float32 datatype supported!");

    // Check channel count
    if (header.channels > 4)
        throw std::runtime_error("Unsupported channel count!");

    // TODO check which dimension is which!?
    m_channels = header.channels;
    m_extent = glm::uvec3(header.size_x, header.size_y, header.size_z);

    std::vector<float> data(glm::prod(m_extent)*m_channels);
    file.read(reinterpret_cast<char*>(data.data()), sizeof(float)*data.size());

    cudaChannelFormatDesc channelDesc;
    switch (header.channels)
    {
        case 1:
        {
            // 1 channel float
            channelDesc = cudaCreateChannelDesc<float1>();
            break;
        }
        case 2:
        {
            // 2 channel float
            channelDesc = cudaCreateChannelDesc<float2>();
            break;
        }
        case 3:
        {
            // 3 channel float
            // TODO 3 channel float is not supported by CUDA!?
            std::vector<float> padded_data(glm::prod(m_extent)*4); // four channels!
            for (uint32_t i = 0; i < glm::prod(m_extent); ++i)
            {
                for (uint32_t j = 0; j < 3; ++j)
                    padded_data[i*4 + j] = data[i*3 + j];
                padded_data[i*4+3] = 0.0f;
            }
            data = std::move(padded_data);
            m_channels = 4;
            channelDesc = cudaCreateChannelDesc<float4>();
            break;
        }
        case 4:
        {
            // 4 channel float
            channelDesc = cudaCreateChannelDesc<float4>();
            break;
        }
        default:
        {
            throw std::runtime_error("Unsupported channel count!");
        }
    }

    cudaExtent cudaExtent { m_extent.x, m_extent.y, m_extent.z };

    CUDA_CHECK( cudaMalloc3DArray(&m_array, &channelDesc, cudaExtent) );

    // TODO this needs to support multi-channel images!
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(data.data(), m_extent.x*m_channels*sizeof(float), m_extent.x, m_extent.y);
    copyParams.dstArray = m_array;
    copyParams.extent = cudaExtent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK( cudaMemcpy3D(&copyParams) );

    cudaResourceDesc resDesc;
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_array;

    cudaTextureDesc texDesc;
    texDesc.addressMode[0] = m_address_mode;
    texDesc.addressMode[1] = m_address_mode;
    texDesc.addressMode[2] = m_address_mode;
    texDesc.filterMode = m_filter_mode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.sRGB = m_srgb ? 1 : 0;
    texDesc.borderColor[0] = m_border_color.x;
    texDesc.borderColor[1] = m_border_color.y;
    texDesc.borderColor[2] = m_border_color.z;
    texDesc.borderColor[3] = m_border_color.w;
    texDesc.normalizedCoords = true; // Always!
    texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.mipmapLevelBias = 0;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = 0;
    texDesc.disableTrilinearOptimization = 0;
#if CUDA_VERSION >= 12000 || (CUDA_VERSION >= 11060 && CUDA_VERSION < 11080)
    texDesc.seamlessCubemap = 0;
#endif

    CUDA_CHECK( cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr) );

    return true;
}

void GridComponent::computeMaximum()
{
    switch (m_channels)
    {
        case 1:
        {
            texture_max_reduce_3d<float>(m_texture, m_extent, &m_maximum);
            break;
        }
        case 2:
        {
            glm::vec2 temp;
            texture_max_reduce_3d<glm::vec2>(m_texture, m_extent, &temp);
            m_maximum = glm::compMax(temp);
            break;
        }
        case 3:
        {
            glm::vec3 temp;
            texture_max_reduce_3d<glm::vec3>(m_texture, m_extent, &temp);
            m_maximum = glm::compMax(temp);
            break;
        }
        case 4:
        {
            glm::vec4 temp;
            texture_max_reduce_3d<glm::vec4>(m_texture, m_extent, &temp);
            m_maximum = glm::compMax(temp);
            break;
        }
        default:
            throw std::runtime_error("Unsupported channel count!");
    }
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY(GridComponent, "grid");

} // end namespace opg
