#include "opg/scene/components/shapes/rectangle.h"

#include "opg/scene/sceneloader.h"

namespace opg {

RectangleShape::RectangleShape(PrivatePtr<Scene> scene, const Properties &_props) :
    MeshShape(std::move(scene), _props, MeshShape::InitInDerivedClass())
{
    const float vertex_data[] = {
        // positions
        -1, -1, 0,  1, -1, 0,
         1,  1, 0, -1,  1, 0,
        // normal
         0, 0, 1, 0, 0, 1,
         0, 0, 1, 0, 0, 1,
        // tangent
         1, 0, 0, 1, 0, 0,
         1, 0, 0, 1, 0, 0,
        // uvs
         0, 0, 1, 0,
         1, 1, 0, 1
    };

    const uint16_t index_data[] = {
        0, 1, 2,  0, 2, 3
    };

    m_vertex_index_buffer.alloc(sizeof(vertex_data) + sizeof(index_data));
    m_vertex_index_buffer.uploadSub(reinterpret_cast<const std::byte*>(vertex_data), sizeof(vertex_data));
    m_vertex_index_buffer.uploadSub(reinterpret_cast<const std::byte*>(index_data), sizeof(index_data), sizeof(vertex_data));

    m_data.indices.data = m_vertex_index_buffer.data() + sizeof(vertex_data);
    m_data.indices.count = 2;
    m_data.indices.byte_stride = sizeof(glm::u16vec3);
    m_data.indices.elmt_byte_size = sizeof(glm::u16vec3);

    m_data.positions.data        = m_vertex_index_buffer.data();
    m_data.positions.count       = 4;
    m_data.positions.byte_stride = sizeof(glm::vec3);

    m_data.normals.data         = m_vertex_index_buffer.data() + sizeof(glm::vec3)*4*1;
    m_data.normals.count        = 4;
    m_data.normals.byte_stride  = sizeof(glm::vec3);

    m_data.tangents.data        = m_vertex_index_buffer.data() + sizeof(glm::vec3)*4*2;
    m_data.tangents.count       = 4;
    m_data.tangents.byte_stride = sizeof(glm::vec3);

    m_data.uvs.data             = m_vertex_index_buffer.data() + sizeof(glm::vec3)*4*3;
    m_data.uvs.count            = 4;
    m_data.uvs.byte_stride      = sizeof(glm::vec2);

    m_data_buffer.upload(&m_data);
}

RectangleShape::~RectangleShape()
{
}



OPG_REGISTER_SCENE_COMPONENT_FACTORY(RectangleShape, "shape.rectangle");

} // namespace opg
