#pragma once

#include "opg/scene/components/shapes/mesh.h"

// #include "opg/scene/components/shapes/rectangle.cuh"

namespace opg {

class RectangleShape : public MeshShape
{
public:
    OPG_API RectangleShape(PrivatePtr<Scene> scene, const Properties &_props);
    virtual ~RectangleShape();

private:
    GenericDeviceBuffer m_vertex_index_buffer;
};

} // end namespace opg
