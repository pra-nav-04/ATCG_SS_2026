#include "radiosityemitter.cuh"

#include "opg/scene/utility/interaction.cuh"

#include <optix.h>

extern "C" __device__ glm::vec3 __direct_callable__evalRadiosity(const SurfaceInteraction &si)
{
    const RadiosityEmitterData *sbt_data = *reinterpret_cast<const RadiosityEmitterData **>(optixGetSbtDataPointer());

    // Ignore backfaces
    if (glm::dot(si.normal, si.incoming_ray_dir) > 0)
        return glm::vec3(0);

    return sbt_data->primitiveRadiosity[si.primitive_index];
}
