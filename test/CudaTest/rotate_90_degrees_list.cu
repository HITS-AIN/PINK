/**
 * @file   CudaTest/rotate_90_degrees_list.cu
 * @date   Nov 14, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "CudaLib/rotate_90_degrees_list.h"

using namespace pink;

TEST(RotationTest, rotate_90_degrees_list)
{
	std::vector<uint32_t> image{1, 2, 3, 4, 0, 0, 0, 0};
    thrust::device_vector<uint32_t> d_image = image;

	dim3 dim_block(1, 1);
	dim3 dim_grid(4, 4, 1);

	rotate_90_degrees_list<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_image[0]), 2, 2, 4);

    thrust::host_vector<uint32_t> result = d_image;
    thrust::host_vector<uint32_t> expected = std::vector<uint32_t>{1, 2, 3, 4, 2, 4, 1, 3};
    EXPECT_EQ(expected, result);
}
