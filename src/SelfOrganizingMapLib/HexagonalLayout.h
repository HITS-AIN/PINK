/**
 * @file   SelfOrganizingMapLib/HexagonalLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstddef>
#include <numeric>

#include "Dimension.h"

namespace pink {

struct HexagonalLayout
{
    static const uint8_t dimensionality = 2;
    static constexpr const char* type = "HexagonalLayout";

    typedef uint32_t IndexType;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;

    auto size() const
    {
        auto dim = 2 * dimension[0] - 1;
        auto radius = (dim - 1)/2;
        return dim * dim - radius * (radius + 1);
    }

    /// Returns the array index of a layout position
    auto get_index(DimensionType const& position) const
    {
    	// proof
        uint32_t index = position[0];
        uint32_t multiplier = dimension[0];
        for (uint8_t i = 1; i < dimensionality; ++i) {
        	index += position[i] * multiplier;
            multiplier *= dimension[i];
        }
        return index;
    }

    /// Returns the layout position of an array index
    auto get_position(IndexType i) const
    {
    	// proof
        int radius = (dimension[0] - 1) / 2;
        int pos = 0;
        for (int x = -radius; x <= radius; ++x) {
            for (int y = -radius - std::min(0, x); y <= radius - std::max(0, x); ++y, ++pos) {
                if (pos == i) return DimensionType({x, y});
            }
        }
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

    DimensionType dimension;
};

} // namespace pink
