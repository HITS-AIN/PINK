/**
 * @file   ImageProcessingLib/circular_euclidean_distance.h
 * @date   Mar 10, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>

#include "ImageProcessingLib/euclidean_distance.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"

namespace pink {

/// Primary template for CircularEuclideanDistanceFunctor
template <typename DataLayout>
struct CircularEuclideanDistanceFunctor
{
    template <typename T>
    T operator () (T const *a, T const *b, DataLayout const& data_layout,
        [[maybe_unused]] uint32_t euclidean_distance_dim) const;
};
/// CircularEuclideanDistanceFunctor: Specialization for CartesianLayout<1>
template <>
struct CircularEuclideanDistanceFunctor<CartesianLayout<1>>
{
    template <typename T>
    T operator () (T const *a, T const *b, CartesianLayout<1> const& data_layout,
        [[maybe_unused]] uint32_t euclidean_distance_dim) const
    {
        return EuclideanDistanceFunctor<CartesianLayout<1>>()(a, b, data_layout, euclidean_distance_dim);
    }
};

/// CircularEuclideanDistanceFunctor: Specialization for CartesianLayout<2>
template <>
struct CircularEuclideanDistanceFunctor<CartesianLayout<2>>
{
    template <typename T>
    T operator () (T const *a, T const *b, CartesianLayout<2> const& data_layout,
        [[maybe_unused]] uint32_t euclidean_distance_dim) const
    {
        T ed = 0;

        auto dim = data_layout.get_dimension(0);
        auto center = dim / 2;
        auto radius = euclidean_distance_dim / 2;
        auto radius_squared = radius * radius;
        auto pa = a;
        auto pb = b;

        for (uint32_t i = 0; i < dim; ++i) {
            for (uint32_t j = 0; j < dim; ++j, ++pa, ++pb) {
                auto dx = i - center;
                auto dy = j - center;
                auto distance_squared = dx * dx + dy * dy;

                if (distance_squared <= radius_squared) {
                    ed += std::pow(*pa - *pb, 2);
                }
            }
        }
        return ed;
    }
};

/// CircularEuclideanDistanceFunctor: Specialization for CartesianLayout<3>
template <>
struct CircularEuclideanDistanceFunctor<CartesianLayout<3>>
{
    template <typename T>
    T operator () (T const *a, T const *b, CartesianLayout<3> const& data_layout,
        [[maybe_unused]] uint32_t euclidean_distance_dim) const
    {
        T ed = 0;

        auto dim = data_layout.get_dimension(0);
        auto radius = dim / 2;
        auto radius_squared = radius * radius;

        for (uint32_t i = 0; i < dim; ++i) {
            for (uint32_t j = 0; j < dim; ++j) {
                auto dx = i - radius;
                auto dy = j - radius;
                auto distance_squared = dx * dx + dy * dy;

                if (distance_squared <= radius_squared) {
                    ed += std::pow(a[i * dim + j] - b[i * dim + j], 2);
                }
            }
        }
        return ed;
    }
};

} // namespace pink
