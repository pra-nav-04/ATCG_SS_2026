#pragma once

#include "opg/preprocessor.h"
#include "opg/glmwrapper.h"

namespace opg {

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;

    OPG_INLINE OPG_HOSTDEVICE glm::vec3 at(float distance) const
    {
        return origin + distance * direction;
    }

    OPG_INLINE OPG_HOSTDEVICE void advance(float distance)
    {
        origin = at(distance);
    }
};

} // namespace opg
