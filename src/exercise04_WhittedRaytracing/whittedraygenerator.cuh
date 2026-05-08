#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"

#include "opg/scene/components/camera.cuh"
#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/utility/trace.cuh"

#include <optix.h>

struct WhittedLaunchParams
{
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

    opg::BufferView<const EmitterVPtrTable *> emitters;

    TraceParameters surface_interaction_trace_params;
    TraceParameters occlusion_trace_params;
    OptixTraversableHandle traversable_handle; // top-level scene IAS
};

#define MAX_TRACE_OPS   (300)
#define MAX_TRACE_DEPTH (8)
#define RAY_STACK_SIZE  (MAX_TRACE_DEPTH+1)
