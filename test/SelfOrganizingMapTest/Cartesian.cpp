/**
 * @file   SelfOrganizingMapTest/Cartesian.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"

#include "SelfOrganizingMapLib/Cartesian.h"
#include "ImageProcessingLib/CropAndRotate.h"

using namespace pink;

TEST(SelfOrganizingMapTest, cartesian_2d)
{
	Cartesian<2, float> c;
	EXPECT_EQ((std::array<uint32_t, 2>{0, 0}), c.get_length());

	Cartesian<2, float> c2({3, 3});
	EXPECT_EQ((std::array<uint32_t, 2>{3, 3}), c2.get_length());

	auto&& rotated_images = CropAndRotate(360)(c2);
}

TEST(SelfOrganizingMapTest, cartesian_2d_cartesian_2d)
{
	Cartesian<2, Cartesian<2, float>> c;
	Cartesian<2, Cartesian<2, float>> c2({3, 3});
}
