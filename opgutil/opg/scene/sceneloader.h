#pragma once

#include <string>
#include <functional>
#include <memory>

#include <pugixml.hpp>

#include "opg/scene/scene.h"
#include "opg/opgapi.h"


#define OPG_REGISTER_SCENE_COMPONENT_FACTORY(Class, Name)   \
    struct RegisterFactory_ ## Class ## _T                  \
    {                                                       \
        RegisterFactory_ ## Class ## _T()                   \
        {                                                   \
            registerSceneComponentFactory(Name,             \
                &Scene::createSceneComponent<Class, const Properties &>);        \
        }                                                   \
        static RegisterFactory_ ## Class ## _T instance;    \
    };                                                      \
    RegisterFactory_ ## Class ## _T RegisterFactory_ ## Class ## _T::instance; \

#define OPG_REGISTER_SCENE_COMPONENT_FACTORY_CUSTOM(Method, Name) \
    struct RegisterFactory_ ## Method ## _T                 \
    {                                                       \
        RegisterFactory_ ## Method ## _T()                  \
        {                                                   \
            registerSceneComponentFactory(Name, Method);    \
        }                                                   \
        static RegisterFactory_ ## Method ## _T instance;   \
    };                                                      \
    RegisterFactory_ ## Method ## _T RegisterFactory_ ## Method ## _T::instance; \

namespace opg {

class SceneComponentRegistry;

typedef std::function<SceneComponent* (Scene *, const Properties &)> SceneComponentFactory;

OPG_API void registerSceneComponentFactory(const std::string &name, SceneComponentFactory factory);


class XMLSceneLoader
{
public:
    OPG_API XMLSceneLoader(Scene *scene);
    OPG_API ~XMLSceneLoader();

    OPG_API void loadFromFile(const std::string &filename);

private:
    SceneComponent *createComponentFromXML(const pugi::xml_node &component_node, const std::string &name_prefix = "");
    void parsePropertyFromXML(const pugi::xml_node &prop_node, Properties &props);
    glm::mat4 parseTransformFromXML(const pugi::xml_node &transform_node);

private:
    Scene *m_scene;
    SceneComponentRegistry *m_registry;
    std::string m_path;
};


} // end namespace opg
