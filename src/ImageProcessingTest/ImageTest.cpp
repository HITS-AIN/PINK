/*
 * ImageProcessingTest.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/ImageIterator.h"
#include "gtest/gtest.h"

using namespace PINK;

TEST(ImageTest, ImageIterator)
{
	typedef ImageIterator<float>::PtrImage PtrImage;

	ImageIterator<float> iterCur("/home/doserbd/cuda-workspace/pink_trunk/RadioGalaxyZoo/code-new/testInput/test.bin");
	PtrImage ptrImage = *iterCur;

	//ptrImage->show();

	EXPECT_EQ(ptrImage->getHeight(), 2);
	EXPECT_EQ(ptrImage->getWidth(), 3);

	std::vector<float> data{1.1, 2.1, 3.1, 4.1, 5.1, 6.1};
	EXPECT_EQ(ptrImage->getPixel(), data);
}
