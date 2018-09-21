/**
 * @file   SelfOrganizingMapTest/Cartesian.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"

#include "SelfOrganizingMapLib/Cartesian.h"

using namespace pink;

TEST(SelfOrganizingMapTest, cartesian_2d)
{
	Cartesian<2, float> c;
	Cartesian<2, float> c2({3,3});
}

TEST(SelfOrganizingMapTest, cartesian_2d_cartesian_2d)
{
	Cartesian<2, Cartesian<2, float>> c;
	Cartesian<2, Cartesian<2, float>> c2({3,3});
}
