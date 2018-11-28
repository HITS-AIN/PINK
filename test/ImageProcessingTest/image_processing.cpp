/**
 * @file   ImageProcessingTest/image_processing.cpp
 * @date   Nov 28, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"
#include <vector>

#include "ImageProcessingLib/rotate.h"

using namespace pink;

TEST(ImageProcessingTest, rotate)
{
    std::vector<float> image1{1, 2, 3, 4};
    std::vector<float> image2(16, 0.0);

    rotate(&image1[0], &image2[0], 2, 2, 4, 4, 2.0 * M_PI, Interpolation::BILINEAR);


}
