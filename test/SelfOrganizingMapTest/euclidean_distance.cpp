/**
 * @file   SelfOrganizingMapTest/euclidean_distance.cpp
 * @date   Jan 29, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

#include "ImageProcessingLib/circular_euclidean_distance.h"
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

TEST(EuclideanDistanceTest, euclidean_distance_cartesian_3d_1)
{
    CartesianLayout<3> layout{2, 2, 2};
    std::vector<int> a{{0,0,0,0,0,0,0,0}};
    std::vector<int> b{{1,1,1,1,1,0,0,1}};

    auto dot = EuclideanDistanceFunctor<CartesianLayout<3>>()(&a[0], &b[0], layout, 2);

    EXPECT_EQ(6, dot);
}

TEST(EuclideanDistanceTest, euclidean_distance_cartesian_3d_2)
{
    CartesianLayout<3> layout{2, 4, 4};
    std::vector<int> a(32, 0);
    std::vector<int> b{{0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0}};

    auto dot = EuclideanDistanceFunctor<CartesianLayout<3>>()(&a[0], &b[0], layout, 2);

    EXPECT_EQ(8, dot);
}

TEST(EuclideanDistanceTest, circular_euclidean_distance_cartesian_2d)
{
    uint32_t dim = 300;
    CartesianLayout<2> layout{dim, dim};
    std::vector<int> a(dim * dim, 0);
    std::vector<int> b(dim * dim, 1);

    auto dot = CircularEuclideanDistanceFunctor<CartesianLayout<2>>()(&a[0], &b[0], layout, 200);

    /// Do you know the number? Isn't it beautiful?
    EXPECT_EQ(31417, dot);
}
