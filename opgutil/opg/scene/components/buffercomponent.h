#pragma once

#include "opg/opgapi.h"
#include "opg/scene/interface/scenecomponent.h"

#include "opg/memory/devicebuffer.h"

#include "opg/opg.h"

namespace opg {

class BufferComponent : public SceneComponent
{
public:
    OPG_API BufferComponent(PrivatePtr<Scene> _scene, GenericDeviceBuffer buffer, const Properties &_props);
    OPG_API BufferComponent(PrivatePtr<Scene> _scene, const Properties &_props);
    virtual ~BufferComponent();

    inline GenericDeviceBuffer &getBuffer() { return m_buffer; }

protected:
    GenericDeviceBuffer m_buffer;
};

OPG_API SceneComponent *createBufferComponentFromFile(Scene *scene, const Properties &props);

} // end namespace opg
