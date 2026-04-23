#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/scenecomponent.h"
#include "opg/scene/components/buffercomponent.h"

#include "opg/memory/bufferview.h"

namespace opg {

class BufferViewComponent : public SceneComponent
{
public:
    OPG_API BufferViewComponent(PrivatePtr<Scene> scene, const Properties &props);
    virtual ~BufferViewComponent();

    inline BufferComponent *getBufferComponent() { return m_buffer_comp; }

    inline const GenericBufferView &getBufferView() { return m_buffer_view; }

protected:
    BufferComponent*    m_buffer_comp;
    GenericBufferView   m_buffer_view;
};

} // end namespace opg
