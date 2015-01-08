/**
 * @file   UtilitiesTest/DistanceFunctorTest.cpp
 * @date   Nov 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/DistanceFunctor.h"
#include "gtest/gtest.h"
#include <cmath>
#include <string>

TEST(DistanceFunctorTest, Cartesian)
{
    // 1-dim
    EXPECT_EQ(CartesianDistanceFunctor<1>(3)(0,0), 0);
    EXPECT_EQ(CartesianDistanceFunctor<1>(3)(0,1), 1);
    EXPECT_EQ(CartesianDistanceFunctor<1>(3)(1,0), 1);
    EXPECT_EQ(CartesianDistanceFunctor<1>(3)(0,2), 2);
    EXPECT_EQ(CartesianDistanceFunctor<1>(3)(2,0), 2);

    // 2-dim
    EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(0,0), 0);
	EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(0,1), 1);
	EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(0,2), 2);
    EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(0,3), 1);
    EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(1,0), 1);
    EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(2,0), 2);
    EXPECT_EQ(CartesianDistanceFunctor<2>(3,3)(3,0), 1);
    EXPECT_FLOAT_EQ(CartesianDistanceFunctor<2>(3,3)(1,3), sqrt(2));
	EXPECT_FLOAT_EQ(CartesianDistanceFunctor<2>(3,3)(1,6), sqrt(5));

    // 3-dim
    EXPECT_EQ(CartesianDistanceFunctor<3>(3,3,3)(0,0), 0);
    EXPECT_EQ(CartesianDistanceFunctor<3>(3,3,3)(0,1), 1);
    EXPECT_EQ(CartesianDistanceFunctor<3>(3,3,3)(0,3), 1);
    EXPECT_EQ(CartesianDistanceFunctor<3>(3,3,3)(3,0), 1);
    EXPECT_EQ(CartesianDistanceFunctor<3>(3,3,3)(0,9), 1);
    EXPECT_FLOAT_EQ((CartesianDistanceFunctor<3>(3,3,3)(13,0)), sqrt(3));

    // 1-dim, periodic
    EXPECT_EQ((CartesianDistanceFunctor<1, true>(3)(0,0)), 0);
    EXPECT_EQ((CartesianDistanceFunctor<1, true>(3)(0,1)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<1, true>(3)(1,0)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<1, true>(3)(0,2)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<1, true>(3)(2,0)), 1);

    // 2-dim, periodic
    EXPECT_EQ((CartesianDistanceFunctor<2, true>(3,3)(0,0)), 0);
    EXPECT_EQ((CartesianDistanceFunctor<2, true>(3,3)(0,1)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<2, true>(3,3)(0,2)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<2, true>(3,3)(3,0)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<2, true>(3,3)(6,0)), 1);
    EXPECT_FLOAT_EQ((CartesianDistanceFunctor<2, true>(3,3)(2,6)), sqrt(2));

    // 3-dim, periodic
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(0,0)), 0);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(0,1)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(0,3)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(3,0)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(0,9)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(0,2)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(0,6)), 1);
    EXPECT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(18,0)), 1);
    EXPECT_FLOAT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(13,0)), sqrt(3));
    EXPECT_FLOAT_EQ((CartesianDistanceFunctor<3, true>(3,3,3)(26,0)), sqrt(3));
}

TEST(DistanceFunctorTest, Hexagonal)
{
	EXPECT_EQ(HexagonalDistanceFunctor(3)(0,1), 1);
	EXPECT_EQ(HexagonalDistanceFunctor(3)(0,2), 1);
	EXPECT_EQ(HexagonalDistanceFunctor(3)(1,2), 2);
    EXPECT_EQ(HexagonalDistanceFunctor(3)(0,3), 1);
    EXPECT_EQ(HexagonalDistanceFunctor(5)(2,3), 3);
}
