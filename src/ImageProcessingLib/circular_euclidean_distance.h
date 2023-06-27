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
        uint32_t euclidean_distance_dim) const
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
        uint32_t euclidean_distance_dim) const
    {
        T ed = 0;

        auto dim = data_layout.get_dimension(0);
        auto center = dim / 2;
        auto radius = euclidean_distance_dim / 2;

        for (uint32_t i = 0; i < euclidean_distance_dim; ++i) {
            auto delta = std::sqrt(2 * radius * (i + 0.5) - std::pow((i + 0.5), 2));
            uint32_t global_i = i + (dim - euclidean_distance_dim) / 2;
            for (uint32_t j = std::round(center - delta); j < std::round(center + delta); ++j) {
                ed += std::pow(a[global_i * dim + j] - b[global_i * dim + j], 2);
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
        uint32_t euclidean_distance_dim) const
    {
        T ed = 0;

        auto depth = data_layout.get_dimension(0);
        auto dim = data_layout.get_dimension(1);
        auto center = dim / 2;
        auto radius = euclidean_distance_dim / 2;

        for (uint32_t d = 0; d < depth; ++d) {
            for (uint32_t i = 0; i < euclidean_distance_dim; ++i) {
                auto delta = std::sqrt(2 * radius * (i + 0.5) - std::pow((i + 0.5), 2));
                uint32_t global_i = i + (dim - euclidean_distance_dim) / 2;
                for (uint32_t j = std::round(center - delta); j < std::round(center + delta); ++j) {
                    ed += std::pow(a[global_i * dim + j] - b[global_i * dim + j], 2);
                }
            }
        }
        return ed;
    }
};

} // namespace pink
