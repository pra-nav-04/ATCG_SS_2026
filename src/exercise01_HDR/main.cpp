#include "opg/opg.h"
#include "opg/imagedata.h"
#include "opg/exception.h"

#include "Robertson.h"
#include "Tonemapping.h"

#include <iostream>
#include <fstream>

void hdr(const std::string &input_folder)
{
    std::string img_txt_path_in = input_folder + "/images.txt";
    std::string output_image_path = input_folder + "/robertson.exr";
    std::string output_crf_path = input_folder + "/output_crf.txt";

    // load data
    std::vector<float> exposures;
    std::vector<opg::ImageData> imgs;

    std::cout << "Read image files:" << std::endl;
    std::ifstream txt_file(img_txt_path_in);
    std::string line;
    while (std::getline(txt_file, line))
    {
        std::istringstream iss(line);
        float exposure;
        std::string path;
        char separator;

        if (!((iss >> exposure >> separator) && std::getline(iss, path, '\0')))
        {
            break; // error
        }

        std::string full_img_path = input_folder + "/" + path;
        std::cout << exposure << " | " << full_img_path << std::endl;
        
        exposures.push_back(exposure);

        opg::ImageData img;
        opg::readImage(full_img_path.c_str(), img);
        // We only accept 3-channel RGB images here.
        OPG_CHECK(img.format == opg::ImageFormat::FORMAT_RGB_UINT8);
        imgs.push_back(img);
    }

    // do processing
    opg::ImageData output_img;
    std::vector<std::vector<float>> output_crfs;
    std::tie(output_img, output_crfs) = robertson(imgs, exposures, 100);

    // write results to disk
    opg::writeImageEXR(output_image_path.c_str(), output_img);

    std::ofstream crf_out(output_crf_path);
    for (size_t i = 0; i < output_crfs[0].size(); i++)
    {
        crf_out << output_crfs[0][i];
        crf_out << ",";
        crf_out << output_crfs[1][i];
        crf_out << ",";
        crf_out << output_crfs[2][i];
        crf_out << std::endl;
    }
    crf_out.close();
}

void tonemap(const std::string &exr_path, const std::string &out_path, const std::string &mode)
{
    std::cout << "tonemap '" << exr_path << "' to '" << out_path << "' with mode '" << mode << "'" << std::endl;
    opg::ImageData img_hdr;
    opg::readImage(exr_path.c_str(), img_hdr);

    opg::ImageData img_ldr = tonemapImage(img_hdr, mode);

    opg::writeImagePNG(out_path.c_str(), img_ldr);
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n"
              << "Options:\n"
              << "  -h | --help                                 Print this usage message\n"
              << "  -d | --dataDir  <data/exercise01_HDR/book>  Data directory\n"
              << "  -m | --mode                                 Tonemapping mode\n"
              << "If no tonemapping mode is specified, the Robertson algorithm is executed.\n"
              << "Valid tonemapping modes are:\n"
              << "- all\n"
              << "- linear_max\n"
              << "- linear_fixed\n"
              << "- gamma_fixed\n"
              << "- histogram\n"
              << std::endl;

    exit(1);
}

int main( int argc, char** argv )
{
    std::string data_dir = opg::getRootPath() + "/data/exercise01_HDR/book";
    std::string tonemapping_mode;

    // parse arguments
    for( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );
        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "-d" || arg == "--dataDir" ) && i + 1 < argc )
        {
            data_dir = argv[++i];
        }
        else if( ( arg == "-m" || arg == "--mode" ) && i + 1 < argc )
        {
            tonemapping_mode = argv[++i];
        }
        else
        {
            std::cerr << "Bad option: '" << arg << "'" << std::endl;
            printUsageAndExit( argv[0] );
        }
    }

    if (tonemapping_mode.empty())
    {
        // use robertson algorithm to reconstruct hdr image from ldr images
        hdr(data_dir);
    }
    else
    {
        // use this to tonemap result from previous task
        // std::string input_exr = "robertson.exr";
        // use this instead to tonemap ground truth image
        std::string input_exr = data_dir + "/hdr_gt.exr";
        std::string output_ldr = data_dir + "/tonemapped_" + tonemapping_mode + ".png";

        // run various tone mapping algorithms
        if (tonemapping_mode == "all")
        {
            tonemap(input_exr, data_dir + "/tonemapped_linear_max.png", "linear_max");
            tonemap(input_exr, data_dir + "/tonemapped_linear_fixed.png", "linear_fixed");
            tonemap(input_exr, data_dir + "/tonemapped_gamma_fixed.png", "gamma_fixed");
            tonemap(input_exr, data_dir + "/tonemapped_histogram.png", "histogram");
        }
        else
        {
            tonemap(input_exr, output_ldr, tonemapping_mode);
        }
    }
}
