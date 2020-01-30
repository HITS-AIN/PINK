/**
 * @file   SelfOrganizingMapTest/generate_rotated_images.cpp
 * @date   Jan 28, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/DataIO.h"
#include "SelfOrganizingMapLib/generate_rotated_images.h"

using namespace pink;

TEST(SelfOrganizingMapTest, generate_rotated_images_2d)
{
    Data<CartesianLayout<2>, int> data{{2, 2}, {1, 2, 3, 4}};
    uint32_t number_of_rotations = 4;
    bool use_flip = false;
    Interpolation interpolation = Interpolation::BILINEAR;
    CartesianLayout<2> neuron_layout{2, 2};

    auto&& spatial_transformed_images = SpatialTransformer<CartesianLayout<2>>()(data, number_of_rotations,
        use_flip, interpolation, neuron_layout);

    EXPECT_EQ((std::vector<int>{{1, 2, 3, 4, 2, 4, 1, 3, 4, 3, 2, 1, 3, 1, 4, 2}}), spatial_transformed_images);
}

TEST(SelfOrganizingMapTest, generate_rotated_images_3d)
{
    Data<CartesianLayout<3>, int> data{{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}};
    uint32_t number_of_rotations = 4;
    bool use_flip = false;
    Interpolation interpolation = Interpolation::BILINEAR;
    CartesianLayout<3> neuron_layout{2, 2, 2};

    auto&& spatial_transformed_images = SpatialTransformer<CartesianLayout<3>>()(data, number_of_rotations,
        use_flip, interpolation, neuron_layout);

    EXPECT_EQ((std::vector<int>{{1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 1, 3, 6, 8, 5, 7, 4, 3, 2, 1, 8, 7, 6, 5, 3, 1, 4, 2, 7, 5, 8, 6}}), spatial_transformed_images);
}
