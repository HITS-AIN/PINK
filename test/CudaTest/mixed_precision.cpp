/**
 * @file   CudaTest/mixed_precision.cpp
 * @date   Apr 16, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#include "CudaLib/CudaLib.h"
#include "CudaLib/euclidean_distance.h"
#include "gtest/gtest.h"
#include <vector>

TEST(mixed_precision, float)
{
	std::vector<float> image{1.0, 2.7, 0.0, -0.8};

    EXPECT_FLOAT_EQ(2.7, image[1]);

    float *d_image = cuda_alloc_float(image.size());
    cuda_copyHostToDevice_float(d_image, &image[0], image.size());

    euclidean_distance(d_image, d_image, image.size());
}
