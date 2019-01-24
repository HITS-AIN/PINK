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

TEST(RotationTest, rotate_and_crop)
{
	int src_height = 1;
	int src_width = 1;
	int dst_height = 3;
	int dst_width = 3;
	float rad = 0.25 * M_PI;

    std::vector<float> src(src_height * src_width, 1.0);
    std::vector<float> dst(dst_height * dst_width, 0.0);

    rotate_and_crop_bilinear(&src[0], &dst[0], src_height, src_width, dst_height, dst_width, rad);

    for (auto&& e : dst) std::cout << e << " ";

}
