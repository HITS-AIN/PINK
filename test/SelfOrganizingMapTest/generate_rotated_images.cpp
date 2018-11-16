/**
 * @file   SelfOrganizingMapTest/generate_rotated_images.cpp
 * @date   Nov 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <vector>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/generate_rotated_images.h"

using namespace pink;

TEST(RotationTest, generate_rotated_images)
{
    Data<CartesianLayout<2>, float> data({2, 2}, std::vector<float>{1,2,3,4});

    auto&& spatial_transformed_images = generate_rotated_images(data, 8, false, Interpolation::BILINEAR, 2);

    //for (auto&& e : spatial_transformed_images) std::cout << e << " ";
}
