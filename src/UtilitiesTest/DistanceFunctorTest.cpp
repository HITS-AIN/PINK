/**
 * @file   UtilitiesTest/DistanceFunctorTest.cpp
 * @date   Nov 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/DistanceFunctor.h"
#include "gtest/gtest.h"
#include <cmath>
#include <string>

TEST(DistanceFunctorTest, Quadratic)
{
	EXPECT_EQ(QuadraticDistanceFunctor()(0,1,3), 1);
	EXPECT_EQ(QuadraticDistanceFunctor()(0,2,3), 2);
	EXPECT_FLOAT_EQ(QuadraticDistanceFunctor()(1,2,3), sqrt(5));
}

TEST(DistanceFunctorTest, Hexagonal)
{
	EXPECT_EQ(HexagonalDistanceFunctor()(0,1,3), 1);
	EXPECT_EQ(HexagonalDistanceFunctor()(0,2,3), 2);
	EXPECT_EQ(HexagonalDistanceFunctor()(1,2,3), 2);
}
