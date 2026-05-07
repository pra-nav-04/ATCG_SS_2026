#pragma once

#include "opg/scene/components/shapeinstance.cuh"

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"

struct MeshShapeData // : public ShapeData
{
    opg::GenericBufferView indices;
    // Per-vertex properties:
    opg::BufferView<glm::vec3> positions;
    opg::BufferView<glm::vec3> normals;
    opg::BufferView<glm::vec3> tangents;
    opg::BufferView<glm::vec2> uvs;
};
