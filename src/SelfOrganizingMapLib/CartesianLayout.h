/**
 * @file   SelfOrganizingMapLib/CartesianLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>

#include "Dimension.h"

namespace pink {

template <uint8_t dim>
struct CartesianLayout
{
    static const uint8_t dimensionality = dim;
    static constexpr const char* type = "CartesianLayout";

    typedef uint32_t IndexType;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;

    auto get_size() const
    {
        return std::accumulate(dimension.begin(), dimension.end(), 1, std::multiplies<uint32_t>());
    }

    /// Returns the array index of a layout position
    auto get_position(DimensionType const& p) const;

    /// Returns the layout position of an array index
    auto get_position(IndexType i) const;

    /// Returns the distance of two neurons given in layout position
    auto get_distance(DimensionType const& p1, DimensionType const& p2) const
    {
    	float distance = 0.0;
    	for (uint8_t i = 0; i < dimensionality; ++i) {
    		distance += std::pow(static_cast<float>(p1[i]) - p2[i], 2);
    	}
        return std::sqrt(distance);
    }

    /// Returns the distance of two neurons given in array indices
    auto get_distance(IndexType i1, IndexType i2) const
    {
        return get_distance(get_position(i1), get_position(i2));
    }

    DimensionType dimension;

};

template <>
inline auto CartesianLayout<1>::get_position(DimensionType const& p) const
{
    return p[0];
}

template <>
inline auto CartesianLayout<2>::get_position(DimensionType const& p) const
{
    return p[0] + p[1] * dimension[0];
}

template <>
inline auto CartesianLayout<3>::get_position(DimensionType const& p) const
{
    return p[0] + p[1] * dimension[0] + p[2] * dimension[0] * dimension[1];
}

template <>
inline auto CartesianLayout<1>::get_position(IndexType i) const
{
    return DimensionType({i});
}

template <>
inline auto CartesianLayout<2>::get_position(IndexType i) const
{
	IndexType y = i / dimension[1];
	IndexType x = i % dimension[1];
    return DimensionType({x, y});
}

template <>
inline auto CartesianLayout<3>::get_position(IndexType i) const
{
	IndexType z = i / dimension[0] / dimension[1];
	IndexType y = (i - z * dimension[0] * dimension[1]) / dimension[1];
	IndexType x = i % dimension[1];
    return DimensionType({x, y, z});
}

} // namespace pink
