/**
 * @file   UtilitiesTest/ipowTest.cpp
 * @date   Jul 31, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>

#include "UtilitiesLib/ipow.h"

using namespace pink;

TEST(ipowTest, ipow)
{
    EXPECT_EQ(256, ipow(2, 8));
    EXPECT_EQ(65536, ipow(2, 16));

    constexpr uint32_t range = ipow(2, std::numeric_limits<uint8_t>::digits) - 1;
    constexpr uint32_t factor = range * range;

    EXPECT_EQ(65025UL, factor);

}
