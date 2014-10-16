/*
 * ImageProcessingTest.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/ImageIterator.h"
#include "gtest/gtest.h"
#include <string>

using namespace PINK;

TEST(ImageTest, ImageIterator)
{
	const std::string filename("image.bin");

	Image<float> image(2,3);
	image.getPixel() = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1};
	image.writeBinary(filename);

	ImageIterator<float> iterCur(filename);

	typedef ImageIterator<float>::PtrImage PtrImage;
	PtrImage ptrImage = *iterCur;

	EXPECT_EQ(ptrImage->getHeight(), 2);
	EXPECT_EQ(ptrImage->getWidth(), 3);

	std::vector<float> data{1.1, 2.1, 3.1, 4.1, 5.1, 6.1};
	EXPECT_EQ(ptrImage->getPixel(), data);
}
