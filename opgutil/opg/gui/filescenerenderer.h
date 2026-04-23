#pragma once

#include <memory>

#include "opg/memory/outputbuffer.h"
#include "opg/memory/devicebuffer.h"
#include "opg/glmwrapper.h"
#include "opg/opgapi.h"

namespace opg {

class Scene;

class FileSceneRenderer
{
public:
    OPG_API FileSceneRenderer(Scene *scene, uint32_t width, uint32_t height);
    OPG_API ~FileSceneRenderer();

    OPG_API void run(const std::string &output_filename, uint32_t sample_count);

private:
    Scene*            m_scene;

    uint32_t          m_width;
    uint32_t          m_height;
};

} // end namespace opg
