/**
 * @file   CudaTest/resize.cu
 * @date   Nov 14, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "CudaLib/resize_kernel.h"

using namespace pink;

TEST(ResizeTest, small)
{
    uint32_t neuron_dim = 2;
    uint32_t image_dim = 4;
    uint32_t min_dim = 2;

    std::vector<uint32_t> image{ 0,  1,  2,  3,
                                 4,  5,  6,  7,
                                 8,  9, 10, 11,
                                12, 13, 14, 15};

    thrust::device_vector<uint32_t> d_image = image;
    thrust::device_vector<uint32_t> d_resized_image(4);

    dim3 dim_block(32, 32);
    dim3 dim_grid(1, 1);

    resize_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_resized_image[0]),
        thrust::raw_pointer_cast(&d_image[0]), neuron_dim, image_dim, min_dim);

    thrust::host_vector<uint32_t> result = d_resized_image;
    thrust::host_vector<uint32_t> expected = std::vector<uint32_t>{5, 6, 9, 10};
    EXPECT_EQ(expected, result);
}
