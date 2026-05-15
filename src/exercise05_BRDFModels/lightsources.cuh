#pragma once

#include "opg/glmwrapper.h"

struct PointLightData
{
    glm::vec3 position;
    glm::vec3 intensity; // W/sr at light source
};

struct DirectionalLightData
{
    glm::vec3 direction;
    glm::vec3 irradiance_at_receiver; // W/m^2 at receiver, dA orthogonal to direction!
};
