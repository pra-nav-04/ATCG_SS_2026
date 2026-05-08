#include "opg/opg.h"

#include "opg/memory/devicebuffer.h"
#include "opg/memory/outputbuffer.h"

#include "opg/glmwrapper.h"

#include "opg/gui/interactivescenerenderer.h"
#include "opg/gui/filescenerenderer.h"

#include "opg/scene/scene.h"
#include "opg/scene/sceneloader.h"

#include <GLFW/glfw3.h>


void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n"
              << "Options:\n"
              << "  -h | --help                           Print this usage message\n"
              << "  -o | --output <output.png>            Output filename (optional)\n"
              << "                                            No GUI will be started if present\n"
              << "  -s | --scene <scene.xml>              Scene to be rendered\n"
              << "  -r | --resolution <width> <height>    Output image width and height\n"
              << std::endl;

    exit(1);
}

int main(int argc, char** argv)
{
    int output_width = 512;
    int output_height = 512;
    std::string output_filename;
    std::string scene_filename;

    // parse arguments
    for( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );
        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "-o" || arg == "--output" ) && i + 1 < argc )
        {
            output_filename = argv[++i];
        }
        else if( ( arg == "-s" || arg == "--scene" ) && i + 1 < argc )
        {
            scene_filename = argv[++i];
        }
        else if( ( arg == "-r" || arg == "--resolution" ) && i + 2 < argc )
        {
            output_width = std::atoi(argv[++i]);
            output_height = std::atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Bad option: '" << arg << "'" << std::endl;
            printUsageAndExit( argv[0] );
        }
    }

    // Set default scene if user did not specify scene
    if (scene_filename.empty())
    {
        std::cerr << "No scene specified" << std::endl;
        printUsageAndExit(argv[0]);
    }

    /*
    // Set default output file prefix
    if (output_file.empty())
    {
        std::cerr << "No output file specified, using default file (output.png)" << std::endl;
        output_file = "output.png";
    }
    */

    // Initialize CUDA
    auto ctx = opg::createOptixContext();

    // Create a scene
    opg::Scene scene(ctx);
    {
        opg::XMLSceneLoader loader(&scene);
        loader.loadFromFile(scene_filename);
    }
    scene.finalize();

    if (output_filename.empty())
    {
        opg::InteractiveSceneRenderer renderer(&scene, output_width, output_height, OPG_TARGET_NAME);
        renderer.run();
    }
    else
    {
        opg::FileSceneRenderer renderer(&scene, output_width, output_height);
        renderer.run(output_filename, 1);
    }

    return 0;
}

