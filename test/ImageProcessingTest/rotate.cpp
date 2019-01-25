/**
 * @file   ImageProcessingLib/rotate.cpp
 * @date   Nov 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "ImageProcessingLib/rotate.h"
#include "UtilitiesLib/EqualFloatArrays.h"

using namespace pink;

TEST(RotationTest, rotate)
{
	int src_height = 1;
	int src_width = 1;
	int dst_height = 3;
	int dst_width = 3;
	float rad = 0.25 * M_PI;

    std::vector<float> src(src_height * src_width, 1.0);
    std::vector<float> dst(dst_height * dst_width, 0.0);

    rotate_bilinear(&src[0], &dst[0], src_height, src_width, dst_height, dst_width, rad);

    std::vector<float> est{0.0, 0.0857864, 0.0, 0.0857864, 1.0, 0.0857864, 0.0, 0.0857864, 0.0};
    EXPECT_TRUE(EqualFloatArrays(est, dst, 1e-4));

}
