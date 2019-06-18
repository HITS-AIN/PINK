/**
 * @file   SelfOrganizingMapTest/Hexagonal.cpp
 * @brief  Unit tests for image processing.
 * @date   Jun 14 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

#include "SelfOrganizingMapLib/HexagonalLayout.h"

using namespace pink;

TEST(HexagonalLayoutTest, hex3)
{
    HexagonalLayout h{{3, 3}};
    EXPECT_EQ(7UL, h.size());

    EXPECT_EQ(0UL, (h.get_index({1, 0})));
    EXPECT_EQ(1UL, (h.get_index({2, 0})));
    EXPECT_EQ(2UL, (h.get_index({0, 1})));
    EXPECT_EQ(3UL, (h.get_index({1, 1})));
    EXPECT_EQ(4UL, (h.get_index({2, 1})));
    EXPECT_EQ(5UL, (h.get_index({0, 2})));
    EXPECT_EQ(6UL, (h.get_index({1, 2})));

    EXPECT_EQ(0.0, h.get_distance({1, 1}, {1, 1}));
    EXPECT_EQ(1.0, h.get_distance({1, 1}, {2, 1}));
    EXPECT_EQ(1.0, h.get_distance({1, 1}, {2, 0}));
    EXPECT_EQ(1.0, h.get_distance({2, 1}, {1, 1}));

    EXPECT_EQ(1.0, h.get_distance({0, 1}, {0, 2}));
    EXPECT_EQ(1.0, h.get_distance({0, 2}, {0, 1}));

    EXPECT_EQ(2.0, h.get_distance({1, 0}, {0, 2}));

    EXPECT_EQ(2.0, h.get_distance(0, 5));
}
