/**
 * @file   CudaTest/RotationTest.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "CudaLib/CudaLib.h"
#include "CudaLib/generate_rotated_images.h"
#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "UtilitiesLib/Filler.h"

using namespace pink;

#if 0
//! Check CUDA image rotation with CPU function.
TEST(RotationTest, rotate_45_degree)
{
    Image<float> image(64, 64, 1.0);
    float angle = 45.0*M_PI/180.0;

    Image<float> image2(64, 64, 0.0);
    //rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image2.getPixel()[0], angle);

    Image<float> image3(64, 64, 0.0);
    cuda_rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image3.getPixel()[0], angle);

    EXPECT_EQ(image2, image3);
}

//! Check CUDA image flipping.
TEST(RotationTest, flip)
{
    int dim = 2;
    int size = dim * dim;
    float *image = new float[size];
    fillWithRandomNumbers(image, size);

    float *d_image = cuda_alloc_float(size);
    cuda_copyHostToDevice_float(d_image, image, size);

    float *image2 = new float[size];
    cuda_copyDeviceToHost_float(image2, d_image, size);

    EXPECT_TRUE(EqualFloatArrays(image, image2, size));

    delete [] image;
    delete [] image2;
}

struct FullRotationTestData
{
    FullRotationTestData(int image_dim, int neuron_dim, int num_rot, bool useFlip,
        Interpolation interpolation, int num_channels)
      : image_dim(image_dim), neuron_dim(neuron_dim), num_rot(num_rot),
        useFlip(useFlip), interpolation(interpolation), num_channels(num_channels)
    {}

    int image_dim;
    int neuron_dim;
    int num_rot;
    bool useFlip;
    Interpolation interpolation;
    int num_channels;
};

class FullRotationTest : public ::testing::TestWithParam<FullRotationTestData>
{};

//! Compare rotated images between CPU and GPU version.
TEST_P(FullRotationTest, generate_rotated_images)
{
    FullRotationTestData data = GetParam();

    int image_size = data.image_dim * data.image_dim;
    int neuron_size = data.neuron_dim * data.neuron_dim;
    int mc_image_size = data.num_channels * image_size;
    int mc_neuron_size = data.num_channels * neuron_size;
    int num_rot_using_flip = data.useFlip ? 2*data.num_rot : data.num_rot;

    std::vector<float> image(mc_image_size);
    std::vector<float> cpu_rotatedImages(num_rot_using_flip * mc_neuron_size);

    fillWithRandomNumbers(image, mc_image_size, 0);

    generateRotatedImages(&cpu_rotatedImages[0], &image[0], data.num_rot, data.image_dim, data.neuron_dim,
        data.useFlip, data.interpolation, data.num_channels);

    thrust::device_vector<float> d_image = image;
    thrust::device_vector<float> d_rotatedImages(num_rot_using_flip * mc_neuron_size, 0.0);

    std::vector<float> cos_alpha(data.num_rot - 1);
    std::vector<float> sin_alpha(data.num_rot - 1);

    float angle_step_radians = 0.5 * M_PI / data.num_rot;
    for (uint32_t i = 0; i < data.num_rot - 1; ++i) {
        float angle = (i+1) * angle_step_radians;
        cos_alpha[i] = std::cos(angle);
        sin_alpha[i] = std::sin(angle);
    }

    thrust::device_vector<float> d_cos_alpha = cos_alpha;
    thrust::device_vector<float> d_sin_alpha = sin_alpha;

    generate_rotated_images(d_rotatedImages, d_image, data.num_rot, data.image_dim, data.neuron_dim,
        data.useFlip, data.interpolation, d_cos_alpha, d_sin_alpha, data.num_channels);

    thrust::host_vector<float> gpu_rotatedImages = d_rotatedImages;

    EXPECT_TRUE(EqualFloatArrays(cpu_rotatedImages, gpu_rotatedImages, num_rot_using_flip * mc_neuron_size));
}

INSTANTIATE_TEST_CASE_P(FullRotationTest_all, FullRotationTest,
    ::testing::Values(
        FullRotationTestData( 3,  3,   4, false, Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 2,  2,   4, false, Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 2,  2,   4, true,  Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 8,  2,   4, false, Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData(64, 44,   4, false, Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData(64, 44,   4, true,  Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData(10, 10,   8, false, Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 4,  2, 360, false, Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 3,  3, 360, true,  Interpolation::NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 4,  2, 360, true,  Interpolation::NEAREST_NEIGHBOR, 1),
//        FullRotationTestData( 2,  2,   8, false, Interpolation::BILINEAR,         1)
//        FullRotationTestData( 4,  2,   4, false, Interpolation::BILINEAR,         1),
        FullRotationTestData( 3,  3,   4, false, Interpolation::NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 3,  3,   8, false, Interpolation::NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 4,  2,   8, false, Interpolation::NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 4,  2,   8, true,  Interpolation::NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 4,  2, 360, true,  Interpolation::NEAREST_NEIGHBOR, 2)
));
#endif
