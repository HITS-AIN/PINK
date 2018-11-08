/**
 * @file   SelfOrganizingMapTest/Cartesian.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "SelfOrganizingMapLib/Data.h"

using namespace pink;

TEST(SelfOrganizingMapTest, data_cartesian_1d)
{
    Data<CartesianLayout<1>, float> c;
    EXPECT_EQ((std::array<uint32_t, 1>{0}), c.get_dimension());

    Data<CartesianLayout<1>, float> c2({2}, std::vector<float>({1, 2}));

    // Check dimension
    EXPECT_EQ((std::array<uint32_t, 1>{2}), c2.get_dimension());

    // Check linear position
    EXPECT_EQ(1, c2[0]);
    EXPECT_EQ(2, c2[1]);

    // Check layout position
    EXPECT_EQ(1, (c2[{0}]));
    EXPECT_EQ(2, (c2[{1}]));
}

TEST(SelfOrganizingMapTest, data_cartesian_2d)
{
    Data<CartesianLayout<2>, float> c;
    EXPECT_EQ((std::array<uint32_t, 2>{0, 0}), c.get_dimension());

    Data<CartesianLayout<2>, float> c2({2, 2}, std::vector<float>({1, 2, 3, 4}));

    // Check dimension
    EXPECT_EQ((std::array<uint32_t, 2>{2, 2}), c2.get_dimension());

    // Check linear position
    EXPECT_EQ(1, c2[0]);
    EXPECT_EQ(2, c2[1]);
    EXPECT_EQ(3, c2[2]);
    EXPECT_EQ(4, c2[3]);

    // Check layout position
    EXPECT_EQ(1, (c2[{0, 0}]));
    EXPECT_EQ(2, (c2[{1, 0}]));
    EXPECT_EQ(3, (c2[{0, 1}]));
    EXPECT_EQ(4, (c2[{1, 1}]));
}

TEST(SelfOrganizingMapTest, data_cartesian_3d)
{
    Data<CartesianLayout<3>, float> c;
    EXPECT_EQ((std::array<uint32_t, 3>{0, 0, 0}), c.get_dimension());

    Data<CartesianLayout<3>, float> c2({2, 2, 2}, std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8}));

    // Check dimension
    EXPECT_EQ((std::array<uint32_t, 3>{2, 2, 2}), c2.get_dimension());

    // Check linear position
    for (int i = 0; i < 8; ++i) EXPECT_EQ(i+1, c2[i]);

    // Check layout position
    EXPECT_EQ(1, (c2[{0, 0, 0}]));
    EXPECT_EQ(2, (c2[{1, 0, 0}]));
    EXPECT_EQ(3, (c2[{0, 1, 0}]));
    EXPECT_EQ(4, (c2[{1, 1, 0}]));
    EXPECT_EQ(5, (c2[{0, 0, 1}]));
    EXPECT_EQ(6, (c2[{1, 0, 1}]));
    EXPECT_EQ(7, (c2[{0, 1, 1}]));
    EXPECT_EQ(8, (c2[{1, 1, 1}]));
}

TEST(SelfOrganizingMapTest, hexagonal_layout)
{
    HexagonalLayout h({3, 3});

    EXPECT_EQ(37UL, h.get_size());
}

TEST(SelfOrganizingMapTest, data_hexagonal)
{
    Data<HexagonalLayout, float> c;
    EXPECT_EQ((std::array<uint32_t, 2>{0, 0}), c.get_dimension());

    Data<HexagonalLayout, float> c2({2, 2}, std::vector<float>({1, 2}));

    // Check dimension
    EXPECT_EQ((std::array<uint32_t, 2>{2, 2}), c2.get_dimension());

    // Check linear position
    for (int i = 0; i < 4; ++i) EXPECT_EQ(i+1, c2[i]);

    // Check layout position
    EXPECT_EQ(1, (c2[{0, 0}]));
    EXPECT_EQ(2, (c2[{1, 0}]));
    EXPECT_EQ(3, (c2[{0, 1}]));
    EXPECT_EQ(4, (c2[{1, 1}]));
}
