/**
 * @file   RotationTest.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "CudaLib/cuda_rotate.h"
#include "gtest/gtest.h"
#include <cmath>

TEST(RotationTest, Basic)
{
	PINK::Image<float> image(64,64,1.0);

	image.show();
	float angle = 45.0*M_PI/180.0;
	PINK::Image<float> image2(64,64);

	cuda_rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image2.getPixel()[0], angle);

	image2.show();
}
