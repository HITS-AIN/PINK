/**
 * @file   ImageProcessingTest/ImageTest.cpp
 * @brief  Unit tests for image class and iterator.
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
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

    EXPECT_EQ(iterCur->getHeight(), 2);
    EXPECT_EQ(iterCur->getWidth(), 3);

    std::vector<float> data{1.1, 2.1, 3.1, 4.1, 5.1, 6.1};
    EXPECT_EQ(iterCur->getPixel(), data);
}
