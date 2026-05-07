#pragma once

#include "opg/scene/components/shapes/mesh.h"

// #include "opg/scene/components/shapes/cube.cuh"

namespace opg {

class CubeShape : public MeshShape
{
public:
    OPG_API CubeShape(PrivatePtr<Scene> scene, const Properties &_props);
    virtual ~CubeShape();

private:
    GenericDeviceBuffer m_vertex_index_buffer;
};

} // end namespace opg
