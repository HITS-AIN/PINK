/**
 * @file   ImageProcessingTest/GenericImageProcessingTest.cpp
 * @brief  Unit tests for image processing.
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "gtest/gtest.h"
#include <vector>

#include "ImageProcessingLib/resize.h"

using namespace pink;

TEST(GenericImageProcessingTest, resize_enlarge)
{
    std::vector<uint8_t> a{1, 2, 3, 4};
    std::vector<uint8_t> b(16, 0);

    resize(&a[0], &b[0], 2, 2, 4, 4);

    EXPECT_EQ(0, b[0]);
    EXPECT_EQ(1, b[5]);
    EXPECT_EQ(4, b[10]);
}

TEST(GenericImageProcessingTest, resize_crop)
{
    std::vector<uint8_t> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<uint8_t> b(4, 0);

    resize(&a[0], &b[0], 4, 4, 2, 2);

    EXPECT_EQ(6, b[0]);
    EXPECT_EQ(7, b[1]);
    EXPECT_EQ(10, b[2]);
    EXPECT_EQ(11, b[3]);
}
