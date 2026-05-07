#include "opg/scene/components/shapes/cube.h"

#include "opg/scene/sceneloader.h"

namespace opg {

CubeShape::CubeShape(PrivatePtr<Scene> scene, const Properties &_props) :
    MeshShape(std::move(scene), _props, MeshShape::InitInDerivedClass())
{
    const float vertex_data[] = {
        // Positions
        // positive X
            1, -1, -1,   1, -1,  1,   1,  1,  1,   1,  1, -1,
        // positive Y
        -1,  1, -1,  -1,  1,  1,   1,  1,  1,   1,  1, -1,
        // positive Z
        -1, -1,  1,  -1,  1,  1,   1,  1,  1,   1, -1,  1,
        // negative X
        -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1,
        // negative Y
        -1, -1, -1,  -1, -1,  1,   1, -1,  1,   1, -1, -1,
        // negative Z
        -1, -1, -1,  -1,  1, -1,   1,  1, -1,   1, -1, -1,

        // Normals
        // positive X
            1,  0,  0,   1,  0,  0,   1,  0,  0,   1,  0,  0,
        // positive Y
            0,  1,  0,   0,  1,  0,   0,  1,  0,   0,  1,  0,
        // positive Z
            0,  0,  1,   0,  0,  1,   0,  0,  1,   0,  0,  1,
        // negative X
        -1,  0,  0,  -1,  0,  0,  -1,  0,  0,  -1,  0,  0,
        // negative Y
            0, -1,  0,   0, -1,  0,   0, -1,  0,   0, -1,  0,
        // negative Z
            0,  0, -1,   0,  0, -1,   0,  0, -1,   0,  0, -1,

        // UVs
        // positive X
        0,  0,  0,  1,  1,  1,  1,  0,
        // positive Y
        0,  0,  0,  1,  1,  1,  1,  0,
        // positive Z
        0,  0,  0,  1,  1,  1,  1,  0,
        // negative X
        0,  0,  0,  1,  1,  1,  1,  0,
        // negative Y
        0,  0,  0,  1,  1,  1,  1,  0,
        // negative Z
        0,  0,  0,  1,  1,  1,  1,  0
    };

    const uint16_t index_data[] = {
        // positive X
        0, 2, 1, 0, 3, 2,
        // positive Y
        4, 5, 6, 4, 6, 7,
        // positive Z
        8, 10, 9, 8, 11, 10,
        // negative X
        12, 13, 14, 12, 14, 15,
        // negative Y
        16, 18, 17, 16, 19, 18,
        // negative Z
        20, 21, 22, 20, 22, 23
    };

    m_vertex_index_buffer.alloc(sizeof(vertex_data) + sizeof(index_data));
    m_vertex_index_buffer.uploadSub(reinterpret_cast<const std::byte*>(vertex_data), sizeof(vertex_data));
    m_vertex_index_buffer.uploadSub(reinterpret_cast<const std::byte*>(index_data), sizeof(index_data), sizeof(vertex_data));

    m_data.indices.data = m_vertex_index_buffer.data() + sizeof(vertex_data);
    m_data.indices.count = 2*6;
    m_data.indices.byte_stride = sizeof(glm::u16vec3);
    m_data.indices.elmt_byte_size = sizeof(glm::u16vec3);

    m_data.positions.data        = m_vertex_index_buffer.data();
    m_data.positions.count       = 6*4;
    m_data.positions.byte_stride = sizeof(glm::vec3);

    m_data.normals.data         = m_vertex_index_buffer.data() + sizeof(glm::vec3)*6*4*1;
    m_data.normals.count        = 6*4;
    m_data.normals.byte_stride  = sizeof(glm::vec3);

    m_data.tangents.data        = m_vertex_index_buffer.data() + sizeof(glm::vec3)*6*4*2;
    m_data.tangents.count       = 6*4;
    m_data.tangents.byte_stride = sizeof(glm::vec3);

    m_data.uvs.data             = m_vertex_index_buffer.data() + sizeof(glm::vec3)*6*4*3;
    m_data.uvs.count            = 6*4;
    m_data.uvs.byte_stride      = sizeof(glm::vec2);

    m_data_buffer.upload(&m_data);
}

CubeShape::~CubeShape()
{
}



OPG_REGISTER_SCENE_COMPONENT_FACTORY(CubeShape, "shape.cube");

} // namespace opg
