/**
 * @file   UtilitiesTest/PointTest.cpp
 * @date   Nov 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/Point.h"
#include "gtest/gtest.h"
#include <cmath>
#include <string>

TEST(PointTest, distance_square)
{
	EXPECT_EQ(distance_square(Point(0,0),Point(0,1)), 1);
	EXPECT_EQ(distance_square(Point(0,0),Point(0,2)), 2);
	EXPECT_FLOAT_EQ(distance_square(Point(1,0),Point(0,2)), sqrt(5));
}

TEST(PointTest, distance_hexagonal)
{
	EXPECT_EQ(distance_hexagonal(Point(0,0),Point(0,1)), 1);
	EXPECT_EQ(distance_hexagonal(Point(0,0),Point(0,2)), 2);
	EXPECT_EQ(distance_hexagonal(Point(1,0),Point(0,2)), 2);
}
