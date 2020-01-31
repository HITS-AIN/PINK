/**
 * @file   ImageProcessingLib/euclidean_distance.h
 * @date   Jan 29, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>

#include "SelfOrganizingMapLib/CartesianLayout.h"

namespace pink {

/// Returns euclidean distance of two arrays without sqrt
template <typename T>
T euclidean_distance(T const* a, T const* b, uint32_t length)
{
    T ed = 0;
    for (uint32_t i = 0; i < length; ++i) { ed += std::pow((*a) - (*b), 2); ++a; ++b; }
    return ed;
}

/// Primary template for EuclideanDistanceFunctor
template <typename DataLayout>
struct EuclideanDistanceFunctor
{
    template <typename T>
    T operator () (T const *a, T const *b, DataLayout const& data_layout,
        uint32_t euclidean_distance_dim) const;
};

/// EuclideanDistanceFunctor: Specialization for CartesianLayout<1>
template <>
struct EuclideanDistanceFunctor<CartesianLayout<1>>
{
    template <typename T>
    T operator () (T const *a, T const *b, CartesianLayout<1> const& data_layout,
        [[maybe_unused]] uint32_t euclidean_distance_dim) const
    {
        assert(euclidean_distance_dim == data_layout.get_dimension(0));
        return euclidean_distance(a, b, data_layout.get_dimension(0));
    }
};

/// EuclideanDistanceFunctor: Specialization for CartesianLayout<2>
template <>
struct EuclideanDistanceFunctor<CartesianLayout<2>>
{
    template <typename T>
    T operator () (T const *a, T const *b, CartesianLayout<2> const& data_layout,
        uint32_t euclidean_distance_dim) const
    {
        T ed = 0;

        auto dim = data_layout.get_dimension(0);
        auto beg = static_cast<uint32_t>((dim - euclidean_distance_dim) * 0.5);
        auto end = beg + euclidean_distance_dim;

        for (uint32_t i = beg; i < end; ++i) {
            for (uint32_t j = beg; j < end; ++j) {
                ed += std::pow(a[i * dim + j] - b[i * dim + j], 2);
            }
        }
        return ed;
    }
};

/// EuclideanDistanceFunctor: Specialization for CartesianLayout<3>
template <>
struct EuclideanDistanceFunctor<CartesianLayout<3>>
{
    template <typename T>
    T operator () (T const *a, T const *b, CartesianLayout<3> const& data_layout,
        uint32_t euclidean_distance_dim) const
    {
        T ed = 0;

        auto dim_d = data_layout.get_dimension(0);
        auto str_d = data_layout.get_stride(0);

        auto dim_i = data_layout.get_dimension(1);
        auto str_i = data_layout.get_stride(1);
        auto beg_i = static_cast<uint32_t>((dim_i - euclidean_distance_dim) * 0.5);
        auto end_i = beg_i + euclidean_distance_dim;

        auto dim_j = data_layout.get_dimension(2);
        auto beg_j = static_cast<uint32_t>((dim_j - euclidean_distance_dim) * 0.5);
        auto end_j = beg_j + euclidean_distance_dim;

        for (uint32_t d = 0; d < dim_d; ++d) {
            for (uint32_t i = beg_i; i < end_i; ++i) {
                for (uint32_t j = beg_j; j < end_j; ++j) {
                    ed += std::pow(a[d * str_d + i * str_i + j] - b[d * str_d + i * str_i + j], 2);
                }
            }
        }
        return ed;
    }
};

} // namespace pink
