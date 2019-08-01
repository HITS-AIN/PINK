/**
 * @file   SelfOrganizingMapTest/Cartesian.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

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
    HexagonalLayout h1({1, 1});
    EXPECT_EQ(1UL, h1.size());

    HexagonalLayout h3({3, 3});
    EXPECT_EQ(7UL, h3.size());

    HexagonalLayout h5({5, 5});
    EXPECT_EQ(19UL, h5.size());

    EXPECT_THROW(HexagonalLayout({0, 0}), pink::exception);
    EXPECT_THROW(HexagonalLayout({1, 0}), pink::exception);
    EXPECT_THROW(HexagonalLayout({2, 2}), pink::exception);
}

TEST(SelfOrganizingMapTest, data_hexagonal)
{
    Data<HexagonalLayout, float> c;

    EXPECT_EQ((std::array<uint32_t, 2>{0, 0}), c.get_dimension());

    Data<HexagonalLayout, float> c2({{3, 3}}, std::vector<float>({0, 1, 2, 3, 4, 5, 6}));

    std::vector<std::pair<uint32_t, uint32_t>> p = {{1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 2}, {1, 2}};

    // Check array index
    for (size_t i = 0; i < p.size(); ++i) {
        EXPECT_EQ(i, (c2[{p[i].first, p[i].second}]));
    }

    // Check layout position
    for (size_t i = 0; i < p.size(); ++i) {
        EXPECT_EQ((std::array<uint32_t, 2>{p[i].first, p[i].second}),
            c2.get_layout().get_position(static_cast<uint32_t>(i)));
    }
}
