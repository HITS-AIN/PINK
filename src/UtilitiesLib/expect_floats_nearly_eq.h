/**
 * @file   CudaTest/expect_floats_nearly_eq.h
 * @date   Apr 9, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>

#define EXPECT_FLOATS_NEARLY_EQ(expected, actual, thresh) \
EXPECT_EQ(expected.size(), actual.size()) << "Array sizes differ.";\
for (size_t idx = 0; idx < std::min(expected.size(), actual.size()); ++idx) \
{ \
    EXPECT_NEAR(expected[idx], actual[idx], thresh) << "at index: " << idx;\
}
