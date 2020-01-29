/**
 * @file   SelfOrganizingMapTest/euclidean_distance.cpp
 * @date   Jan 29, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

#include "ImageProcessingLib/euclidean_distance.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"

using namespace pink;

TEST(EuclideanDistanceTest, euclidean_distance_cartesian_2d_1)
{
    CartesianLayout<2> layout{2, 2};
    std::vector<int> a{{0,0,0,0}};
    std::vector<int> b{{1,0,0,0}};

    auto dot = EuclideanDistanceFunctor<CartesianLayout<2>>()(&a[0], &b[0], layout, 2);

    EXPECT_EQ(1, dot);
}

TEST(EuclideanDistanceTest, euclidean_distance_cartesian_2d_2)
{
    CartesianLayout<2> layout{3, 3};
    std::vector<int> a{{0,0,0,0,0,0,0,0,0}};
    std::vector<int> b{{0,0,1,0,0,1,1,1,1}};

    auto dot = EuclideanDistanceFunctor<CartesianLayout<2>>()(&a[0], &b[0], layout, 2);

    EXPECT_EQ(0, dot);
}

TEST(EuclideanDistanceTest, euclidean_distance_cartesian_2d_4)
{
    CartesianLayout<2> layout{4, 4};
    std::vector<int> a{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
    std::vector<int> b{{1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1}};

    auto dot = EuclideanDistanceFunctor<CartesianLayout<2>>()(&a[0], &b[0], layout, 2);

    EXPECT_EQ(0, dot);
}
