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
    EXPECT_EQ(100, c.size());
}
