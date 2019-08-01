/**
 * @file   SelfOrganizingMapLib/HexagonalLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
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
     : dimension({0, 0}),
       radius(0)
    {}

    HexagonalLayout(DimensionType const& dimension)
     : dimension(dimension),
       radius((dimension[0] - 1) / 2),
       row_size(dimension[0]),
       row_offset(dimension[0] + 1)
    {
        if (dimension[0] % 2 == 0) throw pink::exception("Only odd dimensions are allowed for hexagonal layout");
        if (dimension[0] != dimension[1]) throw pink::exception("dimension[0] must be identical to dimension[1]");

        row_size[radius] = dimension[0];
        for (uint32_t i = 1; i < radius + 1; ++i) {
            row_size[radius + i] = dimension[0] - i;
            row_size[radius - i] = dimension[0] - i;
        }

        row_offset[0] = 0;
        for (size_t i = 0; i < dimension[0]; ++i) {
            row_offset[i + 1] = row_offset[i] + row_size[i];
        }
    }

    bool operator == (SelfType const& other) const
    {
        return dimension == other.dimension;
    }

    /// Returns the number of elements for a hexagonal grid
    auto size() const
    {
        return dimension[0] * dimension[0] - radius * (radius + 1);
    }

    /// Returns the array index of a layout position
    /// position[0] -> q (column index)
    /// position[1] -> r (row index)
    auto get_index(DimensionType const& position) const
    {
        auto index = row_offset[position[1]] + position[0];
        if (radius > position[1]) index -= radius - position[1];
        return index;
    }

    /// Returns the layout position (q, r) of an array index
    auto get_position(IndexType i) const
    {
        uint32_t r = 0;
        for (;r < dimension[0]; ++r) if (i < row_offset[r+1]) break;
        uint32_t q = i - row_offset[r];
        if (radius > r) q += radius - r;
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

    /// Number of rows and columns must be equal and stored in the first element
    DimensionType dimension;

    /// Auxiliary quantity
    uint32_t radius;

    /// Number of elements in a row
    std::vector<uint32_t> row_size;

    /// Starting index of a row
    std::vector<uint32_t> row_offset;
};

} // namespace pink
