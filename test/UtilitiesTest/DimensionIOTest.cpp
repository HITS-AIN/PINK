/**
 * @file   UtilitiesTest/DimensionIOTest.cpp
 * @date   Aug 1, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <sstream>

#include "UtilitiesLib/DimensionIO.h"

using namespace pink;

TEST(DimensionIOTest, dim1)
{
    std::array<uint32_t, 1> d{5};

    std::stringstream ss;
    ss << d;

    EXPECT_EQ("5", ss.str());
}

TEST(DimensionIOTest, dim2)
{
    std::array<uint32_t, 2> d{5, 7};

    std::stringstream ss;
    ss << d;

    EXPECT_EQ("5 x 7", ss.str());
}

TEST(DimensionIOTest, dim3)
{
    std::array<uint32_t, 3> d{5, 7, 3};

    std::stringstream ss;
    ss << d;

    EXPECT_EQ("5 x 7 x 3", ss.str());
}
