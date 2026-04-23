#pragma once

#include "opg/preprocessor.h"
#include "opg/glmwrapper.h"

namespace opg {

// Compute an orthonormal basis where the z-axis corresponds to the localZ vector (in world space).
// The returned matrix acts as "local to world" transformation.
// This is useful when sampling a random direction.
OPG_INLINE OPG_HOSTDEVICE glm::mat3 compute_local_frame(glm::vec3 localZ)
{
    float x  = localZ.x;
    float y  = localZ.y;
    float z  = localZ.z;
    float sz = (z >= 0) ? 1 : -1;
    float a  = 1 / (sz + z);
    float ya = y * a;
    float b  = x * ya;
    float c  = x * sz;

    glm::vec3 localX = glm::vec3(c * x * a - 1, sz * b, c);
    glm::vec3 localY = glm::vec3(b, y * ya - sz, y);

    glm::mat3 frame;
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}


OPG_INLINE OPG_HOSTDEVICE glm::mat3 compute_local_frame_view_normal(glm::vec3 view_dir, glm::vec3 normal)
{
    float NdotV = glm::dot(view_dir, normal);
    if (NdotV > 1 - 1e-6f)
    {
        return compute_local_frame(normal);
    }

    glm::vec3 localZ = normal;
    glm::vec3 localX = glm::normalize(view_dir - NdotV * normal);
    glm::vec3 localY = glm::cross(localZ, localX);

    glm::mat3 frame;
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}

} // namespace opg
