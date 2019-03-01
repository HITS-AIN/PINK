/**
 * @file   ImageProcessingTest/resize.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <vector>

#include "ImageProcessingLib/resize.h"
#include "UtilitiesLib/EqualFloatArrays.h"

using namespace pink;

TEST(GenericImageProcessingTest, resize_enlarge)
{
    std::vector<uint8_t> a{1, 2, 3, 4};
    std::vector<uint8_t> b(16, 0);

    resize(&a[0], &b[0], 2, 2, 4, 4);

    std::vector<uint8_t> est{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT_TRUE(EqualFloatArrays(est, b));
}

TEST(GenericImageProcessingTest, resize_crop)
{
    std::vector<uint8_t> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<uint8_t> b(4, 0);

    resize(&a[0], &b[0], 4, 4, 2, 2);

    std::vector<uint8_t> est{6, 7, 10, 11};
    EXPECT_TRUE(EqualFloatArrays(est, b));
}
