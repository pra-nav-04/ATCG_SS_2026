#include "opg/gui/cameracontroller.h"
#include "opg/scene/components/camera.h"

namespace opg {

void CameraController::wheelEvent(float yscroll)
{
    zoom(m_zoomSpeed * -yscroll);
}

void CameraController::keyEvent(int key, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_RELEASE)
        return;

    for (int i = 0; i < ActionToKeyCount; ++i)
    {
        if (m_keyCodes[i] == key)
            m_keyDown[i] = (action == GLFW_PRESS);
    }
}

void CameraController::tick(float dt)
{
    if (m_keyDown[ActionToKeyIndex::MoveLeft])
        moveLeft(dt*m_moveSpeed*m_lookatDistance);
    if (m_keyDown[ActionToKeyIndex::MoveRight])
        moveRight(dt*m_moveSpeed*m_lookatDistance);

    if (m_keyDown[ActionToKeyIndex::MoveUp])
        moveUp(dt*m_moveSpeed*m_lookatDistance, true);
    if (m_keyDown[ActionToKeyIndex::MoveDown])
        moveDown(dt*m_moveSpeed*m_lookatDistance, true);

    if (m_keyDown[ActionToKeyIndex::MoveForward])
        moveForward(dt*m_moveSpeed*m_lookatDistance, true);
    if (m_keyDown[ActionToKeyIndex::MoveBackward])
        moveBackward(dt*m_moveSpeed*m_lookatDistance, true);
}

void CameraController::resizeViewport(uint32_t width, uint32_t height)
{
    if (!m_camera) return;
    m_camera->setAspectRatio(static_cast<float>(width)/static_cast<float>(height));
}

void CameraController::startTracking(float x, float y)
{
    m_prevPosX = x;
    m_prevPosY = y;
    m_performTracking = true;
}

void CameraController::updateTracking(float x, float y)
{
    if (!m_performTracking)
    {
        startTracking(x, y);
    }

    float deltaX = x - m_prevPosX;
    float deltaY = y - m_prevPosY;

    m_prevPosX = x;
    m_prevPosY = y;

    auto eps = 1e-2f;
    m_polarAngle = glm::clamp(m_polarAngle - m_rotationSpeed*deltaY, eps, glm::pi<float>() - eps);
    m_azimuthAngle = m_azimuthAngle - m_rotationSpeed*deltaX;

    updateCamera();
}

void CameraController::stopTracking(float x, float y)
{
    updateTracking(x, y);
    m_performTracking = false;
}

void CameraController::zoom(float amount)
{
    if (!m_camera) return;
    auto backwardDir = glm::vec3(m_camera->getToWorldMatrix()[2]);

    m_lookatDistance *= glm::exp2(amount);
    m_eye = m_lookAt + m_lookatDistance * backwardDir;

    updateCamera();
}

void CameraController::move(const glm::vec3 &amount)
{
    m_eye += amount;
    m_lookAt += amount;
    updateCamera();
}

void CameraController::moveForward(float amount, bool clip_to_frame)
{
    if (!m_camera) return;
    auto backwardDir = glm::vec3(m_camera->getToWorldMatrix()[2]);
    auto upOfRefDir = glm::vec3(m_referenceFrame[2]);
    if (clip_to_frame)
        backwardDir = glm::normalize(backwardDir - glm::dot(backwardDir, upOfRefDir)*upOfRefDir);
    move(-amount * backwardDir);
}

void CameraController::moveBackward(float amount, bool clip_to_frame)
{
    if (!m_camera) return;
    auto backwardDir = glm::vec3(m_camera->getToWorldMatrix()[2]);
    auto upOfRefDir = glm::vec3(m_referenceFrame[2]);
    if (clip_to_frame)
        backwardDir = glm::normalize(backwardDir - glm::dot(backwardDir, upOfRefDir)*upOfRefDir);
    move(amount * backwardDir);
}

void CameraController::moveLeft(float amount)
{
    if (!m_camera) return;
    auto rightDir = glm::vec3(m_camera->getToWorldMatrix()[0]);
    move(-amount * rightDir);
}

void CameraController::moveRight(float amount)
{
    if (!m_camera) return;
    auto rightDir = glm::vec3(m_camera->getToWorldMatrix()[0]);
    move(amount * rightDir);
}

void CameraController::moveUp(float amount, bool clip_to_frame)
{
    if (!m_camera) return;
    auto upOfRefDir = glm::vec3(m_referenceFrame[2]);
    auto upDir = glm::vec3(m_camera->getToWorldMatrix()[1]);
    if (clip_to_frame)
        upDir = upOfRefDir;
    move(amount * upDir);
}

void CameraController::moveDown(float amount, bool clip_to_frame)
{
    if (!m_camera) return;
    auto upOfRefDir = glm::vec3(m_referenceFrame[2]);
    auto upDir = glm::vec3(m_camera->getToWorldMatrix()[1]);
    if (clip_to_frame)
        upDir = upOfRefDir;
    move(-amount * upDir);
}

void CameraController::rollLeft(float amount)
{
    // TODO not implemented
}

void CameraController::rollRight(float amount)
{
    // TODO not implemented
}

void CameraController::lookAt(const glm::vec3 &center)
{
    // TODO implement
    // glm::lookAt();
}

void CameraController::setReferenceFrame(const glm::mat3 &referenceToWorldFrame)
{
    m_referenceFrame = referenceToWorldFrame;
    // Represent current camera state wrt. new reference frame
    updateState();
}

void CameraController::updateState()
{
    if (m_camera == nullptr)
        return;

    m_lookatDistance = m_camera->getLookatDistance();

    glm::mat4 cameraToWorld = m_camera->getToWorldMatrix();
    glm::vec3 backwardInWorld = glm::vec3(cameraToWorld[2]);

    // TODO assert m_referenceToWorldFrame is rotation matrix!
    glm::mat3 cameraToReference = glm::transpose(m_referenceFrame) * glm::mat3(cameraToWorld);
    glm::vec3 backwardInRef = glm::vec3(cameraToReference[2]);
    glm::vec3 rightInRef = glm::vec3(cameraToReference[0]);
    glm::vec3 upInRef = glm::vec3(cameraToReference[1]);

    m_eye = glm::vec3(cameraToWorld[3]); // = glm::vec3(toWorld * glm::vec4(0, 0, 0, 1))
    m_lookAt = m_eye - backwardInWorld * m_lookatDistance;

    // Only rotation:
    m_polarAngle = glm::acos(backwardInRef.z);
    m_azimuthAngle = glm::atan(backwardInRef.y, backwardInRef.x);
    // NOTE ignoring roll angle!

    updateCamera();
}

void CameraController::updateCamera()
{
    if (m_camera == nullptr)
        return;

    auto backwardInRef = glm::vec3(
        glm::sin(m_polarAngle)*glm::cos(m_azimuthAngle),
        glm::sin(m_polarAngle)*glm::sin(m_azimuthAngle),
        glm::cos(m_polarAngle)
    );
    auto worldUpInRef = glm::vec3(0, 0, 1);
    auto rightInRef = normalize(glm::cross(worldUpInRef, backwardInRef));
    auto upInRef = glm::cross(backwardInRef, rightInRef);

    glm::mat3 cameraToWorldRotation = m_referenceFrame * glm::mat3(rightInRef, upInRef, backwardInRef);
    auto backwardInWorld = cameraToWorldRotation[2];

    if (m_viewMode == EyeFixed)
    {
        m_lookAt = m_eye - backwardInWorld * m_lookatDistance;
    }
    else if (m_viewMode == LookAtFixed)
    {
        m_eye = m_lookAt + backwardInWorld * m_lookatDistance;
    }

    glm::mat4 cameraToWorld = glm::mat4(
        glm::vec4(m_referenceFrame * rightInRef, 0),
        glm::vec4(m_referenceFrame * upInRef, 0),
        glm::vec4(m_referenceFrame * backwardInRef, 0),
        glm::vec4(m_eye, 1)
    );

    m_camera->setToWorldMatrix(cameraToWorld);

    m_camera->setLookatDistance(m_lookatDistance);
}

} // end namespace opg
