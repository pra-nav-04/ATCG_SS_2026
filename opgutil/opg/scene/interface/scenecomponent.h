#pragma once

#include "opg/memory/privateptr.h"
#include "opg/scene/properties.h"

namespace opg {

class Scene;
class RayTracingPipeline;
class ShaderBindingTable;

// This is the base class for all components that live in a Scene.
// This does not only include instances of meshes and cameras, but essentially everything is wrapped in scene components that are created and owned by the scene.
class SceneComponent
{
public:
    SceneComponent(PrivatePtr<Scene> _scene, const Properties &_props) : m_scene{ _scene }, m_name{ _props.getString("name", "") } {}
    virtual ~SceneComponent() {}

    inline Scene *getScene() const { return m_scene; }
    inline const std::string &getName() const { return m_name; }

    // Callbacks invoked during scene/pipeline construction
    inline void ensurePipelineInitialized(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) { if (!m_pipeline_initialized) initializePipeline(pipeline, sbt); m_pipeline_initialized = true; }
    virtual void finalize() {}

    virtual void preLaunchFrame() {}
    virtual void postLaunchFrame() {}

    // Is it worth it to show a GUI node for this component?
    virtual bool hasGui() { return false; }
    // Returns true if the properties of the object have changed and the accumulation buffer should be discarded.
    virtual bool renderGui() { return false; }

    inline uint32_t getRevision() const { return m_revision; }

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) {}

    inline void updateRevision() { ++m_revision; }

protected:
    Scene* m_scene;

private:
    bool m_pipeline_initialized = false;
    std::string m_name;

    // Just a simple counter that can be used to quickly identify changes in the component.
    // The value is modified every time a change is made to the camera data, for example.
    uint32_t m_revision = 0;
};

} // end namespace opg
