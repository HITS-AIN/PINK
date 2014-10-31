/**
 * @file   RotationTest.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "CudaLib/CudaLib.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <vector>

TEST(CudaTest, Rotation)
{
	PINK::Image<float> image(64, 64, 1.0);
	float angle = 45.0*M_PI/180.0;

	PINK::Image<float> image2(64, 64, 0.0);
	//rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image2.getPixel()[0], angle);

	PINK::Image<float> image3(64, 64, 0.0);
	//cuda_rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image3.getPixel()[0], angle);

	EXPECT_EQ(image2,image3);
}

TEST(CudaTest, EuclideanDistance)
{
	int length = 32;
	float *a = new float[length];
	fillRandom(a, length, 0);
	float *b = new float[length];
	fillRandom(b, length, 0);

	float c = calculateEuclideanDistanceWithoutSquareRoot(a, b, length);
	float d = cuda_calculateEuclideanDistanceWithoutSquareRoot(a, b, length);

	std::cout << "c = " << c << std::endl;
	std::cout << "d = " << d << std::endl;

	EXPECT_FLOAT_EQ(c, d);
}
