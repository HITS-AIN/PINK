/**
 * @file   ImageProcessingLib/rotate.cpp
 * @date   Nov 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "ImageProcessingLib/rotate.h"
#include "ImageProcessingLib/rotate_and_crop.h"
#include "UtilitiesLib/EqualFloatArrays.h"

using namespace pink;

template <typename T>
std::ostream& operator << (std::ostream& os, std::vector<T> v)
{
    for (auto e : v) os << e << ", ";
    return os;
}

TEST(RotationTest, rotate)
{
    uint32_t src_height = 1;
    uint32_t src_width = 1;
    uint32_t dst_height = 3;
    uint32_t dst_width = 3;
    float rad = 0.25f * static_cast<float>(M_PI);

    std::vector<float> src(src_height * src_width, 1.0);
    std::vector<float> dst(dst_height * dst_width, 0.0);

    rotate_bilinear(&src[0], &dst[0], src_height, src_width, dst_height, dst_width, rad);

//    std::cout << dst << std::endl;

    std::vector<float> est(dst_height * dst_width, 0.0);
    est[4] = 1.0;
    EXPECT_TRUE(EqualFloatArrays(est, dst, 1e-4f));
}

TEST(RotationTest, rotate2)
{
    uint32_t src_height = 2;
    uint32_t src_width = 2;
    uint32_t dst_height = 3;
    uint32_t dst_width = 3;
    float rad = 0.25f * static_cast<float>(M_PI);

    std::vector<float> src(src_height * src_width, 1.0);
    std::vector<float> dst(dst_height * dst_width, 0.0);

    rotate_bilinear(&src[0], &dst[0], src_height, src_width, dst_height, dst_width, rad);

//    std::cout << dst << std::endl;

    std::vector<float> est(dst_height * dst_width, 0.0);
    est[4] = 1.0;
    EXPECT_TRUE(EqualFloatArrays(est, dst, 1e-4f));
}

TEST(RotationTest, compare_with_crop)
{
    uint32_t src_height = 2;
    uint32_t src_width = 2;
    uint32_t dst_height = 2;
    uint32_t dst_width = 2;
    float rad = 0.0f;

    std::vector<float> src{1.0, 2.0, 3.0, 4.0};
    std::vector<float> dst(dst_height * dst_width, 0.0);
    std::vector<float> dst_crop(dst_height * dst_width, 0.0);

    rotate_bilinear(&src[0], &dst[0], src_height, src_width, dst_height, dst_width, rad);
    rotate_and_crop_bilinear(&src[0], &dst_crop[0], src_height, src_width, dst_height, dst_width, rad);

    EXPECT_TRUE(EqualFloatArrays(dst_crop, dst, 1e-4f));
}
