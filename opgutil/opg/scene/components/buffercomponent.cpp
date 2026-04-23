#include "opg/scene/components/buffercomponent.h"
#include "opg/scene/scene.h"
#include "opg/scene/sceneloader.h"

namespace opg {

BufferComponent::BufferComponent(PrivatePtr<Scene> _scene, GenericDeviceBuffer _buffer, const Properties &_props) :
    m_buffer{ std::move(_buffer) },
    SceneComponent(std::move(_scene), _props)
{
}

BufferComponent::BufferComponent(PrivatePtr<Scene> _scene, const Properties &_props) :
    SceneComponent(std::move(_scene), _props)
{
    if (_props.hasRawData("data"))
    {
        std::vector<char> rawData = _props.getRawData("data").value();
        m_buffer.alloc(rawData.size());
        m_buffer.upload(reinterpret_cast<const std::byte*>(rawData.data()));
    }
    else if (_props.hasFloatData("data"))
    {
        std::vector<float> floatData = _props.getFloatData("data").value();
        m_buffer.alloc(floatData.size() * sizeof(float));
        m_buffer.upload(reinterpret_cast<const std::byte*>(floatData.data()));
    }
}

BufferComponent::~BufferComponent()
{
}


SceneComponent *createBufferComponentFromFile(Scene *scene, const Properties &props)
{
    std::string filename = props.getPath() + props.getString("filename", "");
    std::vector<char> data = readFile(filename.c_str());

    Properties propsCopy = props;
    propsCopy.setRawData("data", std::move(data));
    return scene->createSceneComponent<BufferComponent>(propsCopy);
}

OPG_REGISTER_SCENE_COMPONENT_FACTORY(BufferComponent, "buffer");
OPG_REGISTER_SCENE_COMPONENT_FACTORY_CUSTOM(createBufferComponentFromFile, "filebuffer");

} // end namespace opg
