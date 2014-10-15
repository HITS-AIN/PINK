/*
 * ImageProcessingTest.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "gtest/gtest.h"

extern "C" {
    #include "ImageProcessingLib/ImageProcessing.h"
}

using namespace PINK;

TEST(ImageProcessingTest, Rotation)
{
	ImageIterator<float> iterCur("/home/doserbd/cuda-workspace/pink_trunk/RadioGalaxyZoo/code-new/testInput/boxes.bin");
	Image<float> image = **iterCur;

	//image.show();

	Image<float> image2(image.getHeight(), image.getWidth());
	rotate(image.getHeight(), image.getWidth(), &image.getPixel()[0], &image2.getPixel()[0], 30.0);

	//image2.show();
}
