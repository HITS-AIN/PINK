/**
 * @file   CudaTest/RotationTest.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib/CudaLib.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "UtilitiesLib/Filler.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <vector>

//! Check CUDA image rotation with CPU function.
TEST(RotationTest, 45degree)
{
	PINK::Image<float> image(64, 64, 1.0);
	float angle = 45.0*M_PI/180.0;

	PINK::Image<float> image2(64, 64, 0.0);
	//rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image2.getPixel()[0], angle);

	PINK::Image<float> image3(64, 64, 0.0);
	//cuda_rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image3.getPixel()[0], angle);

	EXPECT_EQ(image2,image3);
}

//! Check CUDA image flipping.
TEST(FlipTest, 0)
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
TEST_P(FullRotationTest, cuda_generateRotatedImages)
{
	FullRotationTestData data = GetParam();

	int image_size = data.image_dim * data.image_dim;
	int neuron_size = data.neuron_dim * data.neuron_dim;
    int mc_image_size = data.num_channels * image_size;
    int mc_neuron_size = data.num_channels * neuron_size;
	int num_rot_using_flip = data.useFlip ? 2*data.num_rot : data.num_rot;

	float *image = new float[mc_image_size];
	float *cpu_rotatedImages = new float[num_rot_using_flip * mc_neuron_size];

	fillWithRandomNumbers(image, mc_image_size, 0);

	generateRotatedImages(cpu_rotatedImages, image, data.num_rot, data.image_dim, data.neuron_dim,
        data.useFlip, data.interpolation, data.num_channels);

//	for (int i = 0; i < data.num_channels * num_rot_using_flip; ++i)
//	    printImage(cpu_rotatedImages + i*neuron_size, data.neuron_dim, data.neuron_dim);
	//writeRotatedImages(cpu_rotatedImages, data.neuron_dim, data.num_rot, "cpu_rot.bin");

	float *d_image = cuda_alloc_float(mc_image_size);
	float *d_rotatedImages = cuda_alloc_float(num_rot_using_flip * mc_neuron_size);
	cuda_fill_zero(d_rotatedImages, num_rot_using_flip * mc_neuron_size);

	cuda_copyHostToDevice_float(d_image, image, mc_image_size);

    // Prepare trigonometric values
	float *d_cosAlpha = NULL, *d_sinAlpha = NULL;
	trigonometricValues(&d_cosAlpha, &d_sinAlpha, data.num_rot/4);

	cuda_generateRotatedImages(d_rotatedImages, d_image, data.num_rot, data.image_dim, data.neuron_dim,
		data.useFlip, data.interpolation, d_cosAlpha, d_sinAlpha, data.num_channels);

	float *gpu_rotatedImages = new float[num_rot_using_flip * mc_neuron_size];
	cuda_copyDeviceToHost_float(gpu_rotatedImages, d_rotatedImages, num_rot_using_flip * mc_neuron_size);

//    for (int i = 0; i < data.num_channels * num_rot_using_flip; ++i)
//	    printImage(gpu_rotatedImages + i*neuron_size, data.neuron_dim, data.neuron_dim);
	//writeRotatedImages(gpu_rotatedImages, data.neuron_dim, data.num_rot, "gpu_rot.bin");

	EXPECT_TRUE(EqualFloatArrays(cpu_rotatedImages, gpu_rotatedImages, num_rot_using_flip * mc_neuron_size));

	cuda_free(d_cosAlpha);
	cuda_free(d_sinAlpha);
	cuda_free(d_image);
	cuda_free(d_rotatedImages);
	delete [] image;
	delete [] cpu_rotatedImages;
	delete [] gpu_rotatedImages;
}

INSTANTIATE_TEST_CASE_P(FullRotationTest_all, FullRotationTest,
    ::testing::Values(
        FullRotationTestData( 3,  3,   4, false, NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 2,  2,   4, false, NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 2,  2,   4, true,  NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 8,  2,   4, false, NEAREST_NEIGHBOR, 1),
        FullRotationTestData(64, 44,   4, false, NEAREST_NEIGHBOR, 1),
        FullRotationTestData(64, 44,   4, true,  NEAREST_NEIGHBOR, 1),
        FullRotationTestData(10, 10,   8, false, NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 4,  2, 360, false, NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 3,  3, 360, true,  NEAREST_NEIGHBOR, 1),
        FullRotationTestData( 4,  2, 360, true,  NEAREST_NEIGHBOR, 1),
//        FullRotationTestData( 2,  2,   8, false, BILINEAR,         1)
//        FullRotationTestData( 4,  2,   4, false, BILINEAR,         1),
        FullRotationTestData( 3,  3,   4, false, NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 3,  3,   8, false, NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 4,  2,   8, false, NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 4,  2,   8, true,  NEAREST_NEIGHBOR, 2),
        FullRotationTestData( 4,  2, 360, true,  NEAREST_NEIGHBOR, 2)
));
