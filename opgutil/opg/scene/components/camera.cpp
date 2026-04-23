#include "opg/scene/components/camera.h"

#include "opg/scene/sceneloader.h"

namespace opg {

Camera::Camera(PrivatePtr<Scene> _scene, const Properties &_props) :
    SceneComponent(std::move(_scene), _props),
    m_toWorld( _props.getMatrix("to_world", glm::mat4(1)) ),
    m_fovY( _props.getFloat("fov_y", M_PI_2) ),
    m_aspectRatio( _props.getFloat("aspect_ratio", 1.0f) ),
    m_lookatDistance( _props.getFloat("lookat_distance", 1.0f) ),
    m_revision( 0 )
{}

Camera::~Camera()
{}


void Camera::getCameraData(CameraData &output_camera) const
{
    // Camera is positioned at origin of local coordinate system
    // output_camera.eye = m_toWorld * glm::vec4(0, 0, 0, 1);
    output_camera.eye = glm::vec3(m_toWorld[3]);

    // output_camera.W = m_toWorld * glm::vec4(0, 0, 1, 0);
    output_camera.U =  glm::normalize(glm::vec3(m_toWorld[0])) * glm::tan(0.5f * m_fovY) * m_aspectRatio;
    output_camera.V = -glm::normalize(glm::vec3(m_toWorld[1])) * glm::tan(0.5f * m_fovY);
    output_camera.W = -glm::normalize(glm::vec3(m_toWorld[2]));
}


OPG_REGISTER_SCENE_COMPONENT_FACTORY(Camera, "camera");

} // end namespace opg
