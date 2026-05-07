#include "opg/scene/components/shapes/mesh.h"
#include "opg/scene/scene.h"
#include "opg/scene/interface/bsdf.h"
#include "opg/scene/interface/emitter.h"
#include "opg/scene/components/bufferviewcomponent.h"

#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"
#include "opg/exception.h"
#include "opg/opg.h"
#include "opg/scene/sceneloader.h"

#include "opg/raytracing/opg_optix_stubs.h"

#include <vector>
#include <map>
#include <tiny_obj_loader.h>


namespace opg {

MeshShape::MeshShape(PrivatePtr<Scene> _scene, const Properties &_props) :
    Shape(std::move(_scene), _props)
{
    if (BufferViewComponent *view_comp = _props.getComponentAs<BufferViewComponent>("indices"))
    {
        m_data.indices   = view_comp->getBufferView();
    }
    if (BufferViewComponent *view_comp = _props.getComponentAs<BufferViewComponent>("positions"))
    {
        m_data.positions = view_comp->getBufferView().asType<glm::vec3>();
    }
    if (BufferViewComponent *view_comp = _props.getComponentAs<BufferViewComponent>("normals"))
    {
        m_data.normals   = view_comp->getBufferView().asType<glm::vec3>();
    }
    if (BufferViewComponent *view_comp = _props.getComponentAs<BufferViewComponent>("tangents"))
    {
        m_data.tangents  = view_comp->getBufferView().asType<glm::vec3>();
    }
    if (BufferViewComponent *view_comp = _props.getComponentAs<BufferViewComponent>("uvs"))
    {
        m_data.uvs       = view_comp->getBufferView().asType<glm::vec2>();
    }

    m_data_buffer.alloc(1);
    m_data_buffer.upload(&m_data);

    m_data_ptr = m_data_buffer.data();
}

MeshShape::MeshShape(PrivatePtr<Scene> _scene, const Properties &_props, InitInDerivedClass) :
    Shape(std::move(_scene), _props)
{
    m_data_buffer.alloc(1);
    m_data_ptr = m_data_buffer.data();
}

MeshShape::~MeshShape()
{
}

void MeshShape::prepareAccelerationStructure()
{
    uint32_t geometry_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE; // OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput build_input = {};
    build_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes         =
        m_data.positions.byte_stride ?
        m_data.positions.byte_stride :
        sizeof(decltype(m_data.positions[0]));
    build_input.triangleArray.numVertices                 = m_data.positions.count;
    build_input.triangleArray.vertexBuffers               = reinterpret_cast<CUdeviceptr*>(&m_data.positions.data);
    build_input.triangleArray.indexFormat                 =
        m_data.indices.elmt_byte_size == sizeof(glm::u16vec3) ?
        OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :  // glm::u16vec3
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;     // glm::u32vec3
    build_input.triangleArray.indexStrideInBytes          = m_data.indices.byte_stride;
    build_input.triangleArray.numIndexTriplets            = m_data.indices.count;
    build_input.triangleArray.indexBuffer                 = reinterpret_cast<CUdeviceptr>(m_data.indices.data);
    build_input.triangleArray.flags                       = &geometry_input_flags;
    build_input.triangleArray.numSbtRecords               = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(m_scene->getContext(), &accel_options, &build_input, 1, &gas_buffer_sizes) );

    GenericDeviceBuffer d_temp(gas_buffer_sizes.tempSizeInBytes);

    m_ast_buffer.alloc(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK( optixAccelBuild(
        m_scene->getContext(), 0,   // CUDA stream
        &accel_options,
        &build_input,
        1,
        d_temp.getRaw(),
        d_temp.byteSize(),
        m_ast_buffer.getRaw(),
        m_ast_buffer.byteSize(),
        &m_ast_handle,
        nullptr,
        0
    ) );
}

void MeshShape::initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt)
{
    auto ptx_filename = getPtxFilename(OPG_TARGET_NAME, "opg/scene/components/shapes/mesh.cu");
    m_hit_prog_group = pipeline->addTrianglesHitGroupShader({ ptx_filename, "__closesthit__mesh" }, {});

    m_sample_position_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__mesh_sample_position" });
    m_eval_position_sampling_pdf_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__mesh_eval_position_sampling_pdf" });
    m_sample_next_event_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__mesh_sample_next_event" });
    m_eval_next_event_sampling_pdf_call_prog_group = pipeline->addCallableShader({ ptx_filename, "__direct_callable__mesh_eval_next_event_sampling_pdf" });
}


SceneComponent *createMeshFromObjFile(Scene *scene, const Properties &props)
{
    std::string filename = props.getPath() + props.getString("filename", "");
    OPG_CHECK(!filename.empty());
    std::string err;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &err, filename.c_str()))
    {
        throw std::runtime_error("Failed to load obj file with error: " + err);
    }


    Properties mesh_props;
    // Copy name
    if (props.hasString("name"))
    {
        mesh_props.setString("name", props.getString("name").value());
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<uint32_t> indices;

    struct less
    {
        bool operator()(const tinyobj::index_t &lhs, const tinyobj::index_t &rhs) const
        {
            if (lhs.vertex_index != rhs.vertex_index)
                return lhs.vertex_index < rhs.vertex_index;
            if (lhs.normal_index != rhs.normal_index)
                return lhs.normal_index < rhs.normal_index;
            return lhs.texcoord_index < rhs.texcoord_index;
        }
    };
    std::map<tinyobj::index_t, uint32_t, less> index_map;

    auto map_index = [&](const tinyobj::index_t &index) -> uint32_t
    {
        auto it = index_map.find(index);
        if (it == index_map.end())
        {
            it = index_map.insert({index, static_cast<uint32_t>(index_map.size())}).first;

            if (!attrib.vertices.empty())
            {
                OPG_CHECK(index.vertex_index >= 0);
                positions.emplace_back(
                    attrib.vertices[3*index.vertex_index+0],
                    attrib.vertices[3*index.vertex_index+1],
                    attrib.vertices[3*index.vertex_index+2]
                );
            }

            if (!attrib.normals.empty())
            {
                OPG_CHECK(index.normal_index >= 0);
                normals.emplace_back(
                    attrib.normals[3*index.normal_index+0],
                    attrib.normals[3*index.normal_index+1],
                    attrib.normals[3*index.normal_index+2]
                );
            }

            if (!attrib.texcoords.empty())
            {
                OPG_CHECK(index.texcoord_index >= 0);
                uvs.emplace_back(
                    attrib.texcoords[2*index.texcoord_index+0],
                    attrib.texcoords[2*index.texcoord_index+1]
                );
            }
        }
        return it->second;
    };


    for (const auto &shape : shapes)
    {
        // If every face has exactly 3 vertices, this holds:
        OPG_CHECK(3*shape.mesh.num_face_vertices.size() == shape.mesh.indices.size());

        for (uint32_t i = 0; i < shape.mesh.indices.size(); ++i)
        {
            uint32_t index = map_index(shape.mesh.indices[i]);
            indices.push_back(index);
        }
    }

    // Add indices to mesh properties
    {
        std::vector<char> raw_indices;
        raw_indices.assign(reinterpret_cast<char*>(indices.data()), reinterpret_cast<char*>(indices.data() + indices.size()));

        auto *buffer_component = scene->createSceneComponent<BufferComponent>(Properties()
                .setRawData("data", raw_indices)
            );

        auto *buffer_view_component = scene->createSceneComponent<BufferViewComponent>(Properties()
                .setComponent("data", buffer_component)
                .setInt("element_size", sizeof(glm::u32vec3))
                .setInt("stride", sizeof(glm::u32vec3))
                // the variable `indices` is a vector of integers, but the buffer view is a vector of integer triplets
                .setInt("count", static_cast<int>(indices.size())/3)
            );

        mesh_props.setComponent("indices", buffer_view_component);
    }

    if (!positions.empty())
    {
        std::vector<char> raw_positions;
        raw_positions.assign(reinterpret_cast<char*>(positions.data()), reinterpret_cast<char*>(positions.data() + positions.size()));
        auto *buffer_component = scene->createSceneComponent<BufferComponent>(Properties()
                .setRawData("data", raw_positions)
            );

        auto *buffer_view_component = scene->createSceneComponent<BufferViewComponent>(Properties()
                .setComponent("data", buffer_component)
                .setInt("element_size", sizeof(glm::vec3))
            );

        mesh_props.setComponent("positions", buffer_view_component);
    }

    if (!normals.empty())
    {
        std::vector<char> raw_normals;
        raw_normals.assign(reinterpret_cast<char*>(normals.data()), reinterpret_cast<char*>(normals.data() + normals.size()));
        auto *buffer_component = scene->createSceneComponent<BufferComponent>(Properties()
                .setRawData("data", raw_normals)
            );

        auto *buffer_view_component = scene->createSceneComponent<BufferViewComponent>(Properties()
                .setComponent("data", buffer_component)
                .setInt("element_size", sizeof(glm::vec3))
            );

        mesh_props.setComponent("normals", buffer_view_component);
    }

    if (!uvs.empty())
    {
        std::vector<char> raw_uvs;
        raw_uvs.assign(reinterpret_cast<char*>(uvs.data()), reinterpret_cast<char*>(uvs.data() + uvs.size()));
        auto *buffer_component = scene->createSceneComponent<BufferComponent>(Properties()
                .setRawData("data", raw_uvs)
            );

        auto *buffer_view_component = scene->createSceneComponent<BufferViewComponent>(Properties()
                .setComponent("data", buffer_component)
                .setInt("element_size", sizeof(glm::vec2))
            );

        mesh_props.setComponent("uvs", buffer_view_component);
    }

    return scene->createSceneComponent<MeshShape>(mesh_props);
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY_CUSTOM(createMeshFromObjFile, "shape.objmesh");
OPG_REGISTER_SCENE_COMPONENT_FACTORY(MeshShape, "shape.mesh");

} // end namespace opg
