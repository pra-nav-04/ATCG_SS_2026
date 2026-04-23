#pragma once

#include <cuda_runtime.h>
#include "opg/glmwrapper.h"
#include "opg/memory/tensorview.h"
#include "opg/memory/bufferview.h"

#include "opg/hostdevice/random.h"
#include "opg/hostdevice/binarysearch.h"

#include <optix.h>

template <typename T = float>
struct GridData
{
    cudaTextureObject_t texture OPG_MEMBER_INIT(0);
    glm::uvec3          extent  OPG_MEMBER_INIT(glm::uvec3(0));
    glm::mat4           to_uvw  OPG_MEMBER_INIT(glm::identity<glm::mat4>());

    //
    // The following code is only valid in device code
    //
    #ifdef __CUDACC__

    __device__ bool is_valid() const
    {
        return texture != 0;
    }

    __device__ T eval(glm::vec3 world_pos) const
    {
        if (!is_valid())
            return T(1);
        glm::vec4 local_pos_hom = to_uvw * glm::vec4(world_pos, 1);
        glm::vec3 local_pos = glm::xyz(local_pos_hom) / glm::www(local_pos_hom);
        T value = tex3D<T>(texture, local_pos);
        return value;
    }

    __device__ glm::vec3 posToLocal(glm::vec3 world_pos) const
    {
        glm::vec4 local_pos_hom = to_uvw * glm::vec4(world_pos, 1);
        glm::vec3 local_pos = glm::xyz(local_pos_hom); // assert w = 1!
        return local_pos;
    }

    __device__ glm::vec3 dirToLocal(glm::vec3 world_dir) const
    {
        glm::vec4 local_dir_hom = to_uvw * glm::vec4(world_dir, 0);
        glm::vec3 local_dir = glm::xyz(local_dir_hom); // assert w=0?
        return local_dir;
    }

    __device__ T evalLocal(glm::vec3 local_pos) const
    {
        if (!is_valid())
            return T(1);

        T value = tex3D<T>(texture, local_pos);
        return value;
    }

    #endif // __CUDACC__
};
