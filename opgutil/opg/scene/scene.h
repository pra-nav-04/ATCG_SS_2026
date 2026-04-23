#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <vector>
#include <map>
#include <functional>

#include "opg/memory/privateptr.h"
#include "opg/raytracing/aabb.h"
#include "opg/raytracing/raytracingpipeline.h"
#include "opg/raytracing/shaderbindingtable.h"

#include "opg/scene/interface/shape.h"
#include "opg/scene/interface/raygenerator.h"
#include "opg/scene/components/shapeinstance.h"

namespace opg {

// The Scene class encapsulates all objects that participate in the rendering of a scene.
class Scene
{
public:
    OPG_API Scene(OptixDeviceContext context);
    OPG_API ~Scene();

    inline OptixDeviceContext getContext() const { return m_context; }

    OPG_API SceneComponent *getSceneComponentByName(const std::string &name);

    template <typename T>
    inline T *getSceneComponentByNameAs(const std::string &name) { return dynamic_cast<T*>(getSceneComponentByName(name)); }

    template <typename T, typename Callback>
    void traverseSceneComponents(Callback callback);

    template <typename T, typename... Args>
    T *createSceneComponent(Args&& ...args);

    OPG_API void finalize();

    OPG_API void traceRays(const TensorView<glm::vec3, 2> &output_buffer);

    OptixTraversableHandle getTraversableHandle(uint32_t rayCount) { return m_ias_handle; }
    opg::Aabb getAABB() const { return m_scene_aabb; }

    inline RayTracingPipeline *getRayTracingPipeline() { return &m_pipeline; }
    inline ShaderBindingTable *getSBT() { return &m_sbt; }

    inline RayGenerator *getRayGenerator() { return m_rayGenerator; }

private:
    OPG_API void addComponent(std::unique_ptr<SceneComponent> component);

    void buildIAS(uint32_t rayCount);

private:
    std::vector<std::unique_ptr<SceneComponent>> m_components;

    std::unordered_map<std::string, SceneComponent*> m_namedComponents;

    std::vector<Shape*>                     m_shapes;
    std::vector<ShapeInstance*>             m_shapeInstances;

    RayGenerator* m_rayGenerator = 0;


    opg::Aabb                                   m_scene_aabb;

    OptixDeviceContext                          m_context = nullptr;

    RayTracingPipeline                          m_pipeline;
    ShaderBindingTable                          m_sbt;

    OptixTraversableHandle                      m_ias_handle = 0;
    GenericDeviceBuffer                         m_ias_buffer;
};


template <typename T, typename... Args>
T *Scene::createSceneComponent(Args&& ...args)
{
    auto uniquePtr = std::make_unique<T>(PrivatePtr<Scene>(this), std::move(args)...);
    auto rawPtr = uniquePtr.get();
    addComponent(std::move(uniquePtr));
    return rawPtr;
}

template <typename T, typename Callback>
void Scene::traverseSceneComponents(Callback callback)
{
    for (auto &component : m_components)
    {
        auto *casted_component = dynamic_cast<T*>(component.get());
        if (casted_component != nullptr)
        {
            callback(casted_component);
        }
    }
}

} // end namespace opg
