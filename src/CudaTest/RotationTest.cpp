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

//! Compare rotated images between CPU and GPU version.
TEST(RotationTest, cuda_generateRotatedImages)
{
	int image_dim = 64;
	int neuron_dim = 44;
	int image_size = image_dim * image_dim;
	int neuron_size = neuron_dim * neuron_dim;
	int num_rot = 2;
	bool flip = false;

	float *image = new float[image_size];
	float *cpu_rotatedImages = new float[num_rot * neuron_size];

	fillRandom(image, image_size, 0);

	generateRotatedImages(cpu_rotatedImages, image, num_rot, image_dim, neuron_dim, flip);

	writeRotatedImages(cpu_rotatedImages, neuron_dim, num_rot, "cpu_rot.bin");
//	for (int i=0; i < num_rot * neuron_size; ++i) {
//		std::cout << "cpu_rotatedImages " << i << ": " << cpu_rotatedImages[i] << std::endl;
//	}

	float *d_image = cuda_alloc_float(image_size);
	float *d_rotatedImages = cuda_alloc_float(num_rot * neuron_size);

	cuda_copyHostToDevice_float(d_image, image, image_size);

    // Prepare trigonometric values
	float angleStepRadians = 2.0 * M_PI / num_rot;

	float angle;
	float *cosAlpha = (float *)malloc(num_rot * sizeof(float));
	float *d_cosAlpha = cuda_alloc_float(num_rot);
	float *sinAlpha = (float *)malloc(num_rot * sizeof(float));
	float *d_sinAlpha = cuda_alloc_float(num_rot);

	for (int i = 0; i < num_rot; ++i) {
		angle = i * angleStepRadians;
	    cosAlpha[i] = cos(angle);
        sinAlpha[i] = sin(angle);
	}

	cuda_copyHostToDevice_float(d_cosAlpha, cosAlpha, num_rot);
	cuda_copyHostToDevice_float(d_sinAlpha, sinAlpha, num_rot);

	cuda_generateRotatedImages(d_rotatedImages, d_image, num_rot, image_dim, neuron_dim, flip, d_cosAlpha, d_sinAlpha);

	float *gpu_rotatedImages = new float[num_rot * neuron_size];
	cuda_copyDeviceToHost_float(gpu_rotatedImages, d_rotatedImages, num_rot * neuron_size);

	writeRotatedImages(gpu_rotatedImages, neuron_dim, num_rot, "gpu_rot.bin");

//	for (int i=0; i < num_rot * neuron_size; ++i) {
//		std::cout << "gpu_rotatedImages " << i << ": " << gpu_rotatedImages[i] << std::endl;
//	}

	EXPECT_TRUE(EqualFloatArrays(cpu_rotatedImages, gpu_rotatedImages, num_rot * neuron_size));

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
