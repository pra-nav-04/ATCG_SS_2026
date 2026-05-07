#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"

struct RadiosityEmitterData
{
    glm::vec3 albedo;
    glm::vec3 emission;
    opg::BufferView<glm::vec3> primitiveRadiosity;
};
