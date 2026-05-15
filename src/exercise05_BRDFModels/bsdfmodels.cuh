#pragma once

#include "opg/scene/interface/bsdf.cuh"

struct OpaqueBSDFData
{
    glm::vec3 diffuse_color;
    glm::vec3 specular_F0;
};

struct RefractiveBSDFData
{
    float index_of_refraction;
};

struct PhongBSDFData
{
    glm::vec3 diffuse_color;
    glm::vec3 specular_F0;
    float exponent;
};

struct WardBSDFData
{
    glm::vec3 diffuse_color;
    glm::vec3 specular_F0;
    float roughness_tangent;
    float roughness_bitangent;
};

struct GGXBSDFData
{
    glm::vec3 diffuse_color;
    glm::vec3 specular_F0;
    float roughness_tangent;
    float roughness_bitangent;
};
