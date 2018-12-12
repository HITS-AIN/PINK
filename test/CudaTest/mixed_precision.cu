/**
 * @file   CudaTest/mixed_precision.cpp
 * @date   Apr 16, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <vector>

#include "CudaLib/dot_dp4a.h"

using namespace pink;

TEST(mixed_precision, dp4a_uint8)
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    if (!(devProp.major >= 6 and devProp.minor >= 1)) {
        // Available after https://github.com/abseil/googletest/pull/1544
        //::testing::GTEST_SKIP();
        // workaround:
        std::cout << "[  SKIPPED ] Feature __dp4a is not supported" << std::endl;
        return;
    }

    std::vector<uint8_t> in1{12, 127, 1, 128};
    std::vector<uint8_t> in2{55, 10, 27, 2};

    uint32_t c_in1 = in1[3];
             c_in1 = c_in1 << 8 | in1[2];
             c_in1 = c_in1 << 8 | in1[1];
             c_in1 = c_in1 << 8 | in1[0];

    uint32_t c_in2 = in2[3];
             c_in2 = c_in2 << 8 | in2[2];
             c_in2 = c_in2 << 8 | in2[1];
             c_in2 = c_in2 << 8 | in2[0];

    thrust::device_vector<uint32_t> d_in1(1, c_in1);
    thrust::device_vector<uint32_t> d_in2(1, c_in2);
    thrust::device_vector<uint32_t> d_in3(1, 0);
    thrust::device_vector<uint32_t> d_out(1, 0);

    dot_dp4a(thrust::raw_pointer_cast(d_in1.data()), thrust::raw_pointer_cast(d_in2.data()),
        thrust::raw_pointer_cast(d_in3.data()), thrust::raw_pointer_cast(d_out.data()));

    EXPECT_EQ(static_cast<uint32_t>(in1[0]*in2[0] + in1[1]*in2[1] + in1[2]*in2[2] + in1[3]*in2[3]), d_out[0]);
}
