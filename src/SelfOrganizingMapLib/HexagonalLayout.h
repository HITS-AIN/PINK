/**
 * @file   SelfOrganizingMapLib/HexagonalLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

#include "UtilitiesLib/DimensionIO.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

struct HexagonalLayout
{
    static const uint8_t dimensionality = 2;
    static constexpr const char* type = "HexagonalLayout";

    typedef uint32_t IndexType;
    typedef HexagonalLayout SelfType;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;

    HexagonalLayout()
     : m_dimension({0, 0}),
       m_radius(0)
    {}

    HexagonalLayout(DimensionType const& dimension)
     : m_dimension(dimension),
       m_radius((dimension[0] - 1) / 2),
       m_row_size(dimension[0]),
       m_row_offset(dimension[0] + 1)
    {
        if (dimension[0] % 2 == 0) throw pink::exception("Only odd dimensions are allowed for hexagonal layout");
        if (dimension[0] != dimension[1]) throw pink::exception("dimension[0] must be identical to dimension[1]");

        m_row_size[m_radius] = dimension[0];
        for (uint32_t i = 1; i < m_radius + 1; ++i) {
            m_row_size[m_radius + i] = dimension[0] - i;
            m_row_size[m_radius - i] = dimension[0] - i;
        }

        m_row_offset[0] = 0;
        for (size_t i = 0; i < dimension[0]; ++i) {
            m_row_offset[i + 1] = m_row_offset[i] + m_row_size[i];
        }
    }

    bool operator == (SelfType const& other) const
    {
        return m_dimension == other.m_dimension;
    }

    /// Returns the number of elements for a hexagonal grid
    auto size() const
    {
        return m_dimension[0] * m_dimension[0] - m_radius * (m_radius + 1);
    }

    static IndexType get_size_from_dim(IndexType dim)
    {
        IndexType radius = (dim - 1) / 2;
        return dim * dim - radius * (radius + 1);
    }

    static IndexType get_dim_from_size(IndexType size)
    {
        auto dim = static_cast<IndexType>(std::sqrt((4 * size - 1) / 3));
        assert(get_size_from_dim(dim) == size);
        return dim;
    }

    /// Returns the array index of a layout position
    /// position[0] -> q (column index)
    /// position[1] -> r (row index)
    auto get_index(DimensionType const& position) const
    {
        auto index = m_row_offset[position[1]] + position[0];
        if (m_radius > position[1]) index -= m_radius - position[1];
        return index;
    }

    /// Returns the layout position (q, r) of an array index
    auto get_position(IndexType i) const
    {
        uint32_t r = 0;
        for (;r < m_dimension[0]; ++r) if (i < m_row_offset[r+1]) break;
        uint32_t q = i - m_row_offset[r];
        if (m_radius > r) q += m_radius - r;
        return DimensionType({q, r});
    }

    /// Returns the distance of two neurons given in layout position
    auto get_distance(DimensionType const& p1, DimensionType const& p2) const
    {
        float distance = 0.0;
        auto dx = static_cast<int32_t>(p1[0]) - static_cast<int32_t>(p2[0]);
        auto dy = static_cast<int32_t>(p1[1]) - static_cast<int32_t>(p2[1]);

        if ((dx >= 0) == (dy >= 0))
            distance = std::abs(dx + dy);
        else
            distance = std::max(std::abs(dx), std::abs(dy));

        return distance;
    }

    /// Returns the distance of two neurons given in array indices
    auto get_distance(IndexType i1, IndexType i2) const
    {
        return get_distance(get_position(i1), get_position(i2));
    }

    auto get_dimension() const { return m_dimension; }

    /// Number of rows and columns must be equal and stored in the first element
    DimensionType m_dimension;

    /// Auxiliary quantity
    uint32_t m_radius;

    /// Number of elements in a row
    std::vector<uint32_t> m_row_size;

    /// Starting index of a row
    std::vector<uint32_t> m_row_offset;
};

} // namespace pink
