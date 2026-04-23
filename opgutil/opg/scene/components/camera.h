#pragma once

#include "opg/scene/interface/scenecomponent.h"
#include "opg/glmwrapper.h"
#include "opg/opgapi.h"

#include "opg/scene/components/camera.cuh"

namespace opg {

class Camera : public SceneComponent
{
public:
    OPG_API Camera(PrivatePtr<Scene> scene, const Properties &props);
    OPG_API virtual ~Camera();

    OPG_API void getCameraData(CameraData &outputData) const;

    inline const glm::mat4 &getToWorldMatrix() const { return m_toWorld; }
    inline void setToWorldMatrix(const glm::mat4 &val) { m_toWorld = val; ++m_revision; }

    inline const float& getFovY() const { return m_fovY; }
    inline void setFovY(float val) { m_fovY = val; ++m_revision; }
    inline const float& getAspectRatio() const { return m_aspectRatio; }
    inline void setAspectRatio(float val) { m_aspectRatio = val; ++m_revision; }

    inline const float& getLookatDistance() const { return m_lookatDistance; }
    inline void setLookatDistance(float val) { m_lookatDistance = val; ++m_revision; }

    inline uint32_t getRevision() const { return m_revision; }

    /*
    Since we want to be able to use glm functions to compute the world transformation,
    the local coordinate system of the camera works as follows:
    X axis = right
    Y axis = up
    Z axis = **backward**

    When creating an image, we want to work in the following coordinate system:
    X axis = right
    Y axis = down
    Z axis = forward
    */

private:
    // View transform
    glm::mat4 m_toWorld;

    // Perspective transform (parameters thereof)
    float m_fovY;
    float m_aspectRatio;
    float m_lookatDistance;

    // Just a simple counter that can be used to identify changes in the camera data.
    // The value is modified every time a change is made to the camera data.
    uint32_t m_revision;
};

} // end namespace opg
