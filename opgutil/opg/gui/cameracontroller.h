#pragma once

#include "opg/scene/components/camera.h"

#include <GLFW/glfw3.h>

namespace opg {

class Camera;

class CameraController
{
public:

    // UI callbacks
    OPG_API void wheelEvent(float yscroll);
    OPG_API void keyEvent(int key, int action, int mods);
    OPG_API void startTracking(float x, float y);
    OPG_API void updateTracking(float x, float y);
    OPG_API void stopTracking(float x, float y);
    OPG_API void tick(float dt);
    OPG_API void resizeViewport(uint32_t width, uint32_t height);

    // Logical commands (i.e. do action)

    OPG_API void zoom(float amount);
    OPG_API void move(const glm::vec3 &amount);
    OPG_API void moveForward(float amount, bool clip_to_frame);
    OPG_API void moveBackward(float amount, bool clip_to_frame);
    OPG_API void moveLeft(float amount);
    OPG_API void moveRight(float amount);
    OPG_API void moveUp(float amount, bool clip_to_frame);
    OPG_API void moveDown(float amount, bool clip_to_frame);
    OPG_API void rollLeft(float amount);
    OPG_API void rollRight(float amount);

    OPG_API void lookAt(const glm::vec3 &center);


    enum ViewMode
    {
        EyeFixed,
        LookAtFixed
    };

    // Initialization

    inline void setViewMode(ViewMode val) { m_viewMode = val; }
    inline ViewMode getViewMode() const { return m_viewMode; }

    inline float getMoveSpeed() const { return m_moveSpeed; }
    inline void setMoveSpeed(const float& val) { m_moveSpeed = val; }

    // Set the camera that will be changed according to user input.
    // Warning, this also initializes the reference frame of the trackball from the camera.
    // The reference frame defines the orbit's singularity.
    OPG_API inline void setCamera(Camera* camera) { m_camera = camera; updateState(); }
    OPG_API inline const Camera* currentCamera() const { return m_camera; }

    // Specify the frame of the orbit that the camera is orbiting around.
    // The important bit is the 'up' of that frame as this is defines the singularity.
    // Here, 'up' is the 'w' component.
    // Typically you want the up of the reference frame to align with the up of the camera.
    // However, to be able to really freely move around, you can also constantly update
    // the reference frame of the trackball. This can be done by calling reinitOrientationFromCamera().
    // In most cases it is not required though (set the frame/up once, leave it as is).
    OPG_API void setReferenceFrame(const glm::mat3 &referenceToWorldFrame);

private:
    // Update controller state based on camera
    OPG_API void updateState();
    // Update camera based on controller state
    OPG_API void updateCamera();

private:
    // The camera controlled by this camera controller
    Camera*      m_camera                   = nullptr;

    // TODO user-tunable parameters
    float        m_zoomSpeed                = 0.5f;
    float        m_moveSpeed                = 1.0f;
    float        m_rotationSpeed            = glm::half_pi<float>() / 180.0f;
    float        m_rollSpeed                = glm::half_pi<float>() / 180.0f;

    // Current state
    glm::vec3    m_eye                      = glm::vec3(0);
    glm::vec3    m_lookAt                   = glm::vec3(0);
    float        m_lookatDistance           = 1.0f;
    float        m_polarAngle               = 0.0f;   // in radians
    float        m_azimuthAngle             = 0.0f;   // in radians

    // Mouse tracking
    ViewMode     m_viewMode                 = LookAtFixed;
    float        m_prevPosX                 = 0.0f;
    float        m_prevPosY                 = 0.0f;
    bool         m_performTracking          = false;

    // Key "tracking"

    enum ActionToKeyIndex
    {
        MoveLeft        = 0,
        MoveRight       = 1,
        MoveUp          = 2,
        MoveDown        = 3,
        MoveForward     = 4,
        MoveBackward    = 5,
        ActionToKeyCount = 6
    };
    int          m_keyCodes[ActionToKeyCount] = { GLFW_KEY_A, GLFW_KEY_D, GLFW_KEY_E, GLFW_KEY_Q, GLFW_KEY_W, GLFW_KEY_S };
    bool         m_keyDown[ActionToKeyCount] = { false };

    // Latitude, longitude etc are wrt. this coordinate system
    glm::mat3    m_referenceFrame           = glm::mat3(
        glm::vec3(0, 0, 1), // "right" of scene in world
        glm::vec3(1, 0, 0), // "forward" of scene in world
        glm::vec3(0, 1, 0)  // "up" of scene in world
    );
};

} // end namespace opg
