#pragma once

#include "opg/glmwrapper.h"

// A structure that holds the relevant information to spwan rays from a perspective camera.
// See also sutil::Camera
struct CameraData
{
    // The positon of the camera in world space.
    glm::vec3               eye;
    // The unnormalized x-axis of the camera coordinate system in world space, defining the vector from the center of the screen to the right edge of the screen.
    glm::vec3               U;
    // The unnormalized y-axis of the camera coordinate system in world space, defining the vector from the center of the screen to the bottom of the screen.
    glm::vec3               V;
    // The unnormalized z-axis of the camera coordinate system in world space, defining the vector from the camera origin towards the screen.
    glm::vec3               W;
};


OPG_INLINE OPG_HOSTDEVICE void spawn_camera_ray(const CameraData &camera_data, const glm::vec2 &viewport_coordinates, glm::vec3 &ray_origin, glm::vec3 &ray_direction)
{
    // Ray starts at eye point
    ray_origin = camera_data.eye;

    // Create a ray on the camera's screen plane (aka near plane)
    ray_direction = glm::normalize(camera_data.U * viewport_coordinates.x + camera_data.V * viewport_coordinates.y + camera_data.W);
    //ray_direction = glm::normalize(glm::vec3(1, 0, 0) * viewport_coordinates.x + glm::vec3(0, 1, 0) * viewport_coordinates.y + glm::vec3(0, 0, 1));
}
