/*
 * ImageProcessingTest.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "gtest/gtest.h"
#include <cmath>

extern "C" {
    #include "ImageProcessingLib/ImageProcessing.h"
}

using namespace PINK;

TEST(ImageProcessingTest, Rotation)
{
	Image<float> image(10,10,1.0);

	//image.show();

	Image<float> image2(image.getHeight(), image.getWidth());
	rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image2.getPixel()[0], 45.0*M_PI/180.0);

	std::vector<float> data{
		0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
		1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
		0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    };

	//image2.show();
	EXPECT_EQ(image2.getPixel(), data);
}
