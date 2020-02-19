/**
 * @file   SelfOrganizingMapLib/CartesianLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>

#include "UtilitiesLib/DimensionIO.h"

namespace pink {

template <uint8_t dim>
struct CartesianLayout
{
    static const uint8_t dimensionality = dim;
    static constexpr const char* type = "CartesianLayout";

    typedef uint32_t IndexType;
    typedef CartesianLayout<dim> SelfType;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;

    bool operator == (SelfType const& other) const
    {
        return m_dimension == other.m_dimension;
    }

    auto size() const
    {
        return std::accumulate(m_dimension.begin(), m_dimension.end(), 1UL, std::multiplies<uint32_t>());
    }

    /// Returns the array index of a layout position
    auto get_index(DimensionType const& p) const;

    /// Returns the layout position of an array index
    auto get_position(IndexType i) const;

    /// Returns the distance of two neurons given in layout position
    auto get_distance(DimensionType const& p1, DimensionType const& p2) const
    {
        float distance = 0.0;
        for (uint8_t i = 0; i < dimensionality; ++i) {
            distance += static_cast<float>(std::pow(static_cast<float>(p1[i]) - p2[i], 2));
        }
        return std::sqrt(distance);
    }

    /// Returns the distance of two neurons given in array indices
    auto get_distance(IndexType i1, IndexType i2) const
    {
        return get_distance(get_position(i1), get_position(i2));
    }

    auto get_dimension() const { return m_dimension; }

    auto get_dimension(IndexType i) const
    {
        assert(i < dim);
        return m_dimension[i];
    }

    auto get_stride(IndexType i) const
    {
        assert(i < dim);
        return std::accumulate(m_dimension.begin() + i + 1, m_dimension.end(), 1UL, std::multiplies<uint32_t>());
    }

    auto get_last_dimension() const { return m_dimension[dimensionality - 1]; }

    auto get_spacing() const;

    DimensionType m_dimension;

};

template <>
inline auto CartesianLayout<1>::get_index(DimensionType const& p) const
{
    return p[0];
}

template <>
inline auto CartesianLayout<2>::get_index(DimensionType const& p) const
{
    return p[0] + p[1] * m_dimension[0];
}

template <>
inline auto CartesianLayout<3>::get_index(DimensionType const& p) const
{
    return p[0] + p[1] * m_dimension[0] + p[2] * m_dimension[0] * m_dimension[1];
}

template <>
inline auto CartesianLayout<1>::get_position(IndexType i) const
{
    return DimensionType({i});
}

template <>
inline auto CartesianLayout<2>::get_position(IndexType i) const
{
    IndexType y = i / m_dimension[1];
    IndexType x = i % m_dimension[1];
    return DimensionType({x, y});
}

template <>
inline auto CartesianLayout<3>::get_position(IndexType i) const
{
    IndexType z = i / m_dimension[0] / m_dimension[1];
    IndexType y = (i - z * m_dimension[0] * m_dimension[1]) / m_dimension[1];
    IndexType x = i % m_dimension[1];
    return DimensionType({x, y, z});
}

template <>
inline auto CartesianLayout<1>::get_spacing() const
{
    return 1;
}

template <>
inline auto CartesianLayout<2>::get_spacing() const
{
    return 1;
}

template <>
inline auto CartesianLayout<3>::get_spacing() const
{
    return m_dimension[0];
}

} // namespace pink
