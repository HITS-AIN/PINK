/**
 * @file   SelfOrganizingMapTest/Cartesian.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

#include "SelfOrganizingMapLib/CartesianLayout.h"

using namespace pink;

TEST(CartesianLayoutTest, cartesian_2d)
{
    CartesianLayout<2> c{10, 10};
    EXPECT_EQ(100UL, c.size());

    EXPECT_EQ(0.0f, c.get_distance({0, 0}, {0, 0}));
    EXPECT_EQ(1.0f, c.get_distance({0, 0}, {0, 1}));
    EXPECT_EQ(2.0f, c.get_distance({0, 0}, {0, 2}));
    EXPECT_NEAR(static_cast<float>(std::sqrt(2.0)), c.get_distance({0, 0}, {1, 1}), 1e-7f);
    EXPECT_NEAR(static_cast<float>(std::sqrt(8.0)), c.get_distance({0, 0}, {2, 2}), 1e-7f);
}

TEST(CartesianLayoutTest, cartesian_3d)
{
    CartesianLayout<3> c{2, 3, 4};

    EXPECT_EQ(24UL, c.size());
    EXPECT_EQ(12UL, c.get_stride(0));
    EXPECT_EQ(4UL, c.get_stride(1));
    EXPECT_EQ(1UL, c.get_stride(2));
}
