#pragma once

#include <memory>

#include "opg/scene/interface/scenecomponent.h"
#include "opg/glmwrapper.h"
#include "opg/memory/tensorview.h"
#include "opg/memory/devicebuffer.h"

#include "opg/scene/components/camera.h"

namespace opg {

// Each scene must contain exactly one RayGenerator component.
// This component is responsible for rendering images via the launchFrame function that is invoked from the scene itself.
class RayGenerator : public SceneComponent
{
public:
    RayGenerator(PrivatePtr<Scene> _scene, const Properties &_props) :
        SceneComponent(std::move(_scene), _props),
        m_camera{ _props.getComponentAs<Camera>("camera") }
    {}
    virtual ~RayGenerator() {}

    // TODO add ray generator specific methods!
    //virtual void resizeOutput(uint32_t width, uint32_t height);

    virtual void launchFrame(CUstream stream, const TensorView<glm::vec3, 2> &output_buffer) = 0;

    inline Camera *getCamera() const { return m_camera; }
    inline void setCamera(Camera *_camera) { m_camera = _camera; }

    // TODO not just ray generator but all components?
    virtual void onSceneChanged() {}

protected:
    virtual void initializePipeline(RayTracingPipeline *pipeline, ShaderBindingTable *sbt) override = 0; // this must be overriden by children

protected:
    // Active camera used by this ray generator.
    // We placed this in the interface since all our ray generators somehow rely on a camera!
    Camera *m_camera;
};

} // end namespace opg
