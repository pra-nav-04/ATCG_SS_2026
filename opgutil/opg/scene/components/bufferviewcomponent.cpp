#include "opg/scene/components/bufferviewcomponent.h"

#include "opg/scene/sceneloader.h"

namespace opg {

BufferViewComponent::BufferViewComponent(PrivatePtr<Scene> _scene, const Properties &_props) :
    SceneComponent(std::move(_scene), _props),
    m_buffer_comp{ _props.getComponentAs<BufferComponent>("data") }
{
    uint32_t offset = _props.getInt("offset", 0);

    uint32_t element_size = _props.getInt("element_size", 0);
    OPG_ASSERT_MSG(element_size > 0, "Element size in buffer view must be greater than 0!");

    uint32_t stride = _props.getInt("stride", 0);
    if (stride == 0)
    {
        // Assume data to be tightly packed
        stride = element_size;
    }

    uint32_t count = _props.getInt("count", 0);
    if (count == 0)
    {
        // Assume that we have a view into the whole buffer
        count = (static_cast<uint32_t>(m_buffer_comp->getBuffer().size()) - offset) / stride;
    }

    m_buffer_view.data = m_buffer_comp->getBuffer().data() + offset;
    m_buffer_view.count = count;
    m_buffer_view.byte_stride = stride;
    m_buffer_view.elmt_byte_size = element_size;
}

BufferViewComponent::~BufferViewComponent()
{
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY(BufferViewComponent, "bufferview");

} // end namespace opg
