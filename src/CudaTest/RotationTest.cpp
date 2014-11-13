/**
 * @file   CudaTest/RotationTest.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib/CudaLib.h"
#include "EqualFloatArrays.h"
#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
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
	int image_dim = 2;
	int image_size = image_dim * image_dim;
	float *image = new float[image_size];
	float *flippedImage = new float[image_size];

	EXPECT_TRUE(EqualFloatArrays(image,flippedImage,image_size));

	delete [] image;
	delete [] flippedImage;
}

struct FullRotationTestData
{
	FullRotationTestData(int image_dim, int neuron_dim, int num_rot, bool useFlip)
	  : image_dim(image_dim), neuron_dim(neuron_dim), num_rot(num_rot), useFlip(useFlip)
	{}

	int image_dim;
	int neuron_dim;
	int num_rot;
	bool useFlip;
};

class FullRotationTest : public ::testing::TestWithParam<FullRotationTestData>
{};

//! Compare rotated images between CPU and GPU version.
TEST_P(FullRotationTest, cuda_generateRotatedImages)
{
	FullRotationTestData data = GetParam();

	int image_size = data.image_dim * data.image_dim;
	int neuron_size = data.neuron_dim * data.neuron_dim;
	int num_rot_using_flip = data.useFlip ? 2*data.num_rot : data.num_rot;
	int image_size_using_flip = data.useFlip ? 2*image_size : image_size;

	float *image = new float[image_size];
	float *cpu_rotatedImages = new float[num_rot_using_flip * neuron_size];

	fillRandom(image, image_size, 0);

	generateRotatedImages(cpu_rotatedImages, image, data.num_rot, data.image_dim, data.neuron_dim, data.useFlip);

	//writeRotatedImages(cpu_rotatedImages, neuron_dim, num_rot, "cpu_rot.bin");

	float *d_image = cuda_alloc_float(image_size_using_flip);
	float *d_rotatedImages = cuda_alloc_float(num_rot_using_flip * neuron_size);

	cuda_copyHostToDevice_float(d_image, image, image_size);

    // Prepare trigonometric values
	float angleStepRadians = 2.0 * M_PI / data.num_rot;

	float angle;
	float *cosAlpha = (float *)malloc(data.num_rot * sizeof(float));
	float *d_cosAlpha = cuda_alloc_float(data.num_rot);
	float *sinAlpha = (float *)malloc(data.num_rot * sizeof(float));
	float *d_sinAlpha = cuda_alloc_float(data.num_rot);

	for (int i = 0; i < data.num_rot-1; ++i) {
		angle = (i+1) * angleStepRadians;
	    cosAlpha[i] = cos(angle);
        sinAlpha[i] = sin(angle);
	}

	cuda_copyHostToDevice_float(d_cosAlpha, cosAlpha, data.num_rot);
	cuda_copyHostToDevice_float(d_sinAlpha, sinAlpha, data.num_rot);

	cuda_generateRotatedImages(d_rotatedImages, d_image, data.num_rot, data.image_dim, data.neuron_dim,
		data.useFlip, d_cosAlpha, d_sinAlpha);

	float *gpu_rotatedImages = new float[num_rot_using_flip * neuron_size];
	cuda_copyDeviceToHost_float(gpu_rotatedImages, d_rotatedImages, num_rot_using_flip * neuron_size);

	//writeRotatedImages(gpu_rotatedImages, neuron_dim, num_rot, "gpu_rot.bin");

	EXPECT_TRUE(EqualFloatArrays(cpu_rotatedImages, gpu_rotatedImages, num_rot_using_flip * neuron_size));

	cuda_free(d_cosAlpha);
	cuda_free(d_sinAlpha);
	cuda_free(d_image);
	cuda_free(d_rotatedImages);
	delete [] cosAlpha;
	delete [] sinAlpha;
	delete [] image;
	delete [] cpu_rotatedImages;
	delete [] gpu_rotatedImages;
}

INSTANTIATE_TEST_CASE_P(FullRotationTest_all, FullRotationTest,
    ::testing::Values(
        FullRotationTestData(3,3,2,false),
        //FullRotationTestData(2,2,2,true),
        FullRotationTestData(64,44,2,false)
        //FullRotationTestData(64,44,2,true)
));
