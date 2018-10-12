/**
 * @file   SelfOrganizingMapTest/Cartesian.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "ImageProcessingLib/CropAndRotate.h"

using namespace pink;

TEST(SelfOrganizingMapTest, cartesian_2d)
{
    Data<CartesianLayout<2>, float> c;
    EXPECT_EQ((std::array<uint32_t, 2>{0, 0}), c.get_dimension());

    Data<CartesianLayout<2>, float> c2({3, 3});
    EXPECT_EQ((std::array<uint32_t, 2>{3, 3}), c2.get_dimension());

    //auto&& rotated_images = CropAndRotate(360)(c2);
}
