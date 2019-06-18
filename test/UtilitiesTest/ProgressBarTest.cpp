/**
 * @file   UtilitiesTest/ProgressBarTest.cpp
 * @date   Jun 13, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

#include "UtilitiesLib/ProgressBar.h"

using namespace pink;

TEST(ProgressBarTest, valid)
{
    std::stringstream ss;

    ProgressBar progress(100, 70, 10, ss);

    EXPECT_FALSE(progress.valid());
    EXPECT_EQ(ss.str(), "");
    for (int i = 0; i < 10; ++i) ++progress;
    EXPECT_TRUE(progress.valid());
    EXPECT_EQ(ss.str(), "[=======>                                                              ] 10 % 0 s\n");
}

TEST(ProgressBarTest, invalid_construction)
{
    EXPECT_THROW(ProgressBar(0, 0, 0), pink::exception);

    EXPECT_THROW(ProgressBar(-10,  10,  10), pink::exception);
    EXPECT_THROW(ProgressBar( 10, -10,  10), pink::exception);
    EXPECT_THROW(ProgressBar( 10,  10, -10), pink::exception);

    EXPECT_THROW(ProgressBar(100, 9, 10), pink::exception);
}
