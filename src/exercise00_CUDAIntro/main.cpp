#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "opg/opg.h"
#include "opg/exception.h"
#include "opg/imagedata.h"
#include "opg/memory/devicebuffer.h"
#include "opg/glmwrapper.h"

#include "kernels.h"

void taskA()
{
    /* Task:
     * Using the GLM math library:
     * - Consider the equation $<n, x> + d = 0$ in 3d space that holds for all points on a plane, where n is a unit-length vector and d is a scalar.
     *   Determine the parameters n, d that describes the plane spanned by the given plane points.
     * - Swap the y and z components of the plane normal.
     * - Find the intersection point between the *new* plane (modified n, old d) and the ray.
     *    - Note: the ray parameters are given as CUDA-builtin float3 vectors.
     *    - Convert the ray parameters to glm::vec3 using the cuda2glm() function.
     *    - Implement the ray-plane intersection.
     *    - Convert the intersection point back to a CUDA-builtin float3 vector using the glm2cuda() function.
     */

    std::cout << "taskA output:" << std::endl;

    // Plane equiation: dot(n, x) + d = 0
    glm::vec3 plane_point_1 = glm::vec3(3, 0, 2);
    glm::vec3 plane_point_2 = glm::vec3(4, 3, 2);
    glm::vec3 plane_point_3 = glm::vec3(1, 2, 4);

    glm::vec3 plane_tangent_1 = plane_point_2 - plane_point_1;
    glm::vec3 plane_tangent_2 = plane_point_3 - plane_point_1;
    glm::vec3 plane_normal = glm::normalize(glm::cross(plane_tangent_1, plane_tangent_2));
    float plane_d = -glm::dot(plane_normal, plane_point_1);

    std::cout << "\tplane normal:             " << plane_normal << ", plane offset: " << plane_d << std::endl;

    std::swap(plane_normal.y, plane_normal.z);

    std::cout << "\tswizzled plane normal:    " << plane_normal << std::endl;

    // Ray equation: x = origin + t * dir
    float3 ray_origin_cudafloat3 = make_float3(1, 4, 2);
    float3 ray_dir_cudafloat3 = make_float3(2, 4, 3);

    glm::vec3 ray_origin = cuda2glm(ray_origin_cudafloat3);
    glm::vec3 ray_dir = cuda2glm(ray_dir_cudafloat3);
    float denominator = glm::dot(plane_normal, ray_dir);
    OPG_CHECK(std::abs(denominator) > 1e-6f);
    float ray_t = -(glm::dot(plane_normal, ray_origin) + plane_d) / denominator;
    glm::vec3 intersection = ray_origin + ray_t * ray_dir;
    float3 intersection_cudafloat3 = glm2cuda(intersection);

    std::cout << "\tray-plane intersection at (" << intersection_cudafloat3.x << " " << intersection_cudafloat3.y << " " << intersection_cudafloat3.z << ")" << std::endl;
}

void taskB()
{
    /* Task:
     * - Generate an array containing integer numbers 1..1e7 and copy it to the GPU.
     * - Use a CUDA kernel to multiply each number by a constant in parallel.
     * - Copy the result back to the CPU host memory.
     */

    constexpr int totalCount = 10000000;
    constexpr int multiplier = 2;

    std::vector<int> dataArray(totalCount);
    opg::DeviceBuffer<int> d_dataArray(totalCount);

    launchGenerateSequence(d_dataArray.data(), totalCount);
    launchMultiplyIntArray(d_dataArray.data(), totalCount, multiplier);
    CUDA_SYNC_CHECK();
    d_dataArray.download(dataArray.data());

    std::cout << "taskB output:" << std::endl;
    std::cout << "\tfirst 10 entries:";
    for (int i = 0; i < std::min<int>(dataArray.size(), 10); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
    std::cout << "\tlast 10 entries:";
    for (int i = std::max<int>(dataArray.size()-10, 0); i < dataArray.size(); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
}

void taskC()
{
    /* Task:
     * - Apply the separable Sobel filter to an image.
     *   Use two kernel executions to compute a single 2D convolution.
     * - Output G = sqrt(G_x^2 + G_y^2)
     */

    opg::ImageData imageData;
    // opg::getRootPath returns the absolute path to the root directory
    // of this framework independent of the current working directory
    std::string filename = opg::getRootPath() + "/data/exercise00_CUDAIntro/Valve.png";
    OPG_CHECK(opg::readImage(filename.c_str(), imageData));
    uint32_t channelSize = opg::getImageFormatChannelSize(imageData.format);
    uint32_t channelCount = opg::getImageFormatChannelCount(imageData.format);

    // Each color channel stores a single uint8_t (or unsigned char) value per pixel.
    OPG_CHECK(channelSize == 1);

    const int width = static_cast<int>(imageData.width);
    const int height = static_cast<int>(imageData.height);
    const int channels = static_cast<int>(channelCount);
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * channels;

    opg::DeviceBuffer<uint8_t> d_input(pixelCount);
    opg::DeviceBuffer<uint8_t> d_output(pixelCount);
    opg::DeviceBuffer<float> d_temp(pixelCount);
    opg::DeviceBuffer<float> d_gradX(pixelCount);
    opg::DeviceBuffer<float> d_gradY(pixelCount);

    d_input.upload(imageData.data.data());

    launchSobelDerivativeXPass1(d_input.data(), d_temp.data(), width, height, channels);
    CUDA_SYNC_CHECK();
    launchSobelDerivativeXPass2(d_temp.data(), d_gradX.data(), width, height, channels);
    CUDA_SYNC_CHECK();

    launchSobelDerivativeYPass1(d_input.data(), d_temp.data(), width, height, channels);
    CUDA_SYNC_CHECK();
    launchSobelDerivativeYPass2(d_temp.data(), d_gradY.data(), width, height, channels);
    CUDA_SYNC_CHECK();

    launchSobelMagnitude(d_input.data(), d_gradX.data(), d_gradY.data(), d_output.data(), width, height, channels);
    CUDA_SYNC_CHECK();

    d_output.download(imageData.data.data());

    // Write your solution back into the imageData.data array such that we can safe it as an image again
    opg::writeImagePNG("taskC_output.png", imageData);
}

void taskD()
{
    /* Task:
     * - Create an array of uniformly distributed (pseudo) random floats in [0, 1] on the GPU.
     *   You can use tea<4>(a, b) to initialize a random number generator, and rnd(seed) to generate a pseudo random number on the GPU. (See sutil/cuda/random.h)
     * - Count all entries that are greater than 0.5 in parallel.
     */

    float threshold = 0.5f;
    int totalCount = 10000000;
    int aboveThresholdCount = 0; // compute this value

    opg::DeviceBuffer<float> d_randomValues(totalCount);
    opg::DeviceBuffer<int> d_aboveThresholdCount(1);
    CUDA_CHECK(cudaMemset(d_aboveThresholdCount.data(), 0, sizeof(int)));

    launchGenerateRandomAndCount(d_randomValues.data(), totalCount, threshold, d_aboveThresholdCount.data());
    CUDA_SYNC_CHECK();
    d_aboveThresholdCount.download(&aboveThresholdCount);

    std::cout << "taskD output:" << std::endl;
    std::cout << "\t" << aboveThresholdCount << "/" << totalCount << " values are greater than " << threshold << std::endl;
}

void taskE()
{
    /* Task:
     * - Implement a matrix multiplication in a CUDA kernel.
     *   Store your matrices in row major order (https://en.wikipedia.org/wiki/Row-_and_column-major_order).
     */

    int lhsRows = 4;
    int lhsCols = 10;
    std::vector<float> lhs = { // 4x10
        97, 95, 80, 31, 31, 72,  1,  2, 88, 93,
        46, 58, 54, 94, 84, 59, 75,  4, 78, 62,
        44, 92, 14, 83, 82, 47, 78, 88, 28, 12,
        22, 96, 61, 93, 95, 77, 27, 35, 58, 53
    };
    int rhsRows = 10;
    int rhsCols = 3;
    std::vector<float> rhs = { // 10x3
        79, 64, 67,
        67, 14, 43,
        68, 17, 43,
        75,  4,  3,
        94, 81, 83,
         1, 56, 18,
        52, 59,  0,
        88, 89, 96,
        93, 66, 76,
        38, 33, 12
    };
    int outputRows = 4;
    int outputCols = 3;
    std::vector<float> output;
    output.resize(outputRows * outputCols);
    // Expected output:
    // 36725, 24679, 25982,
    // 40059, 27133, 23270,
    // 39432, 28626, 26127,
    // 40192, 26453, 26179

    OPG_CHECK(lhsCols == rhsRows);
    OPG_CHECK(outputRows == lhsRows);
    OPG_CHECK(outputCols == rhsCols);

    opg::DeviceBuffer<float> d_lhs(lhs.size());
    opg::DeviceBuffer<float> d_rhs(rhs.size());
    opg::DeviceBuffer<float> d_output(output.size());

    d_lhs.upload(lhs.data());
    d_rhs.upload(rhs.data());

    launchMatrixMultiply(d_lhs.data(), d_rhs.data(), d_output.data(), lhsRows, lhsCols, rhsCols);
    CUDA_SYNC_CHECK();
    d_output.download(output.data());

    std::cout << "taskE output:" << std::endl;
    for (int r = 0; r < outputRows; ++r)
    {
        std::cout << "\t";
        for (int c = 0; c < outputCols; ++c)
        {
            std::cout << " " << output[r * outputCols + c];
        }
        std::cout << std::endl;
    }
}

int main()
{
    taskA();
    taskB();
    taskC();
    taskD();
    taskE();
}
