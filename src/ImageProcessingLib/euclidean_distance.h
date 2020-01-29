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
        uint32_t euclidean_distance_dim) const
	{
	    assert(euclidean_distance_dim == data_layout.m_dimension[0]);
        return euclidean_distance(a, b, data_layout.m_dimension[0]);
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

		auto dim = data_layout.m_dimension[0];
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

		auto dim = data_layout.m_dimension[0];
		auto beg = static_cast<uint32_t>((dim - euclidean_distance_dim) * 0.5);
		auto end = beg + euclidean_distance_dim;

		for (uint32_t i = beg; i < end; ++i) {
			for (uint32_t j = beg; j < end; ++j) {
				ed += euclidean_distance(a + (i * data_layout.m_dimension[1] + j) * data_layout.m_dimension[2],
                                         b + (i * data_layout.m_dimension[1] + j) * data_layout.m_dimension[2],
										 data_layout.m_dimension[2]);
			}
		}
		return ed;
	}
};

} // namespace pink
