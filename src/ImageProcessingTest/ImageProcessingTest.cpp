/*
 * ImageProcessingTest.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "gtest/gtest.h"
#include <cmath>
#include "ImageProcessingLib/ImageProcessing.h"


using namespace PINK;

TEST(ImageProcessingTest, Rotation)
{
	Image<float> image(10,10,1.0);

	//image.show();

	Image<float> image2(image.getHeight(), image.getWidth());
	rotate(image.getHeight(), image.getWidth(), image.getPointerOfFirstPixel(), image2.getPointerOfFirstPixel(), 45.0*M_PI/180.0, NONE);

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

TEST(ImageProcessingTest, EuclideanSimilarity)
{
	std::vector<float> a{2.0, -3.9, 0.1};
	std::vector<float> b{1.9, -4.0, 0.2};

	EXPECT_NEAR(0.1732 ,(calculateEuclideanSimilarity(&a[0], &b[0], a.size())), 1e-4);
}
