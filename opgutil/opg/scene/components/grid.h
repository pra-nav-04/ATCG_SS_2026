#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/components/bufferviewcomponent.h"

#include "opg/opg.h"

#include "opg/scene/components/grid.cuh"

namespace opg {

class GridComponent : public SceneComponent
{
public:
    OPG_API GridComponent(PrivatePtr<Scene> _scene, const Properties &_props);
    virtual ~GridComponent();

    inline float getMaximum() { return m_maximum; }
    inline glm::uvec3 getExtent() const { return m_extent; }

    template <typename T=float>
    GridData<T> getGridData() const;

protected:
    bool tryLoadNpy(const std::string &filename);
    bool tryLoadVol(const std::string &filename);

    void computeMaximum();

protected:
    cudaArray_t         m_array;
    cudaTextureObject_t m_texture;
    glm::uvec3          m_extent;
    uint32_t            m_channels;

    bool                m_srgb;
    glm::vec4           m_border_color;
    cudaTextureAddressMode m_address_mode;
    cudaTextureFilterMode  m_filter_mode;

    float               m_maximum;

    glm::mat4           m_to_uvw;
};

template <typename T>
GridData<T> GridComponent::getGridData() const
{
    // TODO check if m_channels is compatible with type T!
    //if (m_channels != channel_count<T>::value)
    //    throw std::runtime_error("Incompatible grid type!");
    GridData<T> grid_data;
    grid_data.extent = m_extent;
    grid_data.texture = m_texture;
    grid_data.to_uvw = m_to_uvw;
    return grid_data;
}

} // end namespace opg
