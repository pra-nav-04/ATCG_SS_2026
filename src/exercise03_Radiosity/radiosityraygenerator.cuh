#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"

#include "opg/scene/components/camera.cuh"
#include "opg/scene/utility/trace.cuh"

#include <optix.h>

struct ComputeFormFactorMatrixInstanceData
{
    opg::GenericBufferView      indices;                    // The index buffer defining the vertex indices of each triangle primitive
    opg::BufferView<glm::vec3>  positions;                  // The vertex positions in object space
    opg::BufferView<glm::vec3>  normals;                    // The vertex normals in object space
    glm::mat4                   transform;                  // A matrix transforming vertices from object space to world space
    int                         form_factor_matrix_offset;  // Offset into the form factor matrix of the first triangle in this mesh
};

struct RadiosityLaunchParams
{
    // Parameters used during rendering of the radiosity solution to the screen

    // Small floating point number used to move rays away from surfaces when tracing to avoid self intersections.
    float scene_epsilon;

    // 2D output buffer storing the final linear color values.
    // Each thread in the __raygen__main shader program computes one pixel value in this array.
    // Access via output_buffer[y][x].value() or output_buffer(glm::uvec2(x, y)).value()
    opg::TensorView<glm::vec3, 2> output_radiance;

    // Size of the image we are generating
    uint32_t image_width;
    uint32_t image_height;

    CameraData camera;


    // Parameters used during generation of the form factor matrix
    float*              form_factor_matrix;      // form_factor_matrix stores a row-major (data is laid out as a sequence of rows in memory) and tighly packed square matrix.
    uint32_t            form_factor_matrix_size; // form_factor_matrix is square, so only one size parameter is needed!

    ComputeFormFactorMatrixInstanceData instance_1;
    ComputeFormFactorMatrixInstanceData instance_2;


    // Common parameters
    TraceParameters surface_interaction_trace_params;
    TraceParameters occlusion_trace_params;
    OptixTraversableHandle traversable_handle; // top-level scene IAS
};
