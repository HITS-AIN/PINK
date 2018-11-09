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

    auto get_size() const
    {
        auto dim_square = 2 * dimension[0] + 1;
        return dim_square * dim_square - dimension[0] * (dimension[0] + 1);
    }

    auto get_position(DimensionType const& position) const
    {
        uint32_t linear_position = position[0];
        uint32_t multiplier = dimension[0];
        for (size_t i = 1; i < dimensionality; ++i) {
            linear_position += position[i] * multiplier;
            multiplier *= dimension[i];
        }
        return linear_position;
    }

    DimensionType dimension;
};

} // namespace pink
