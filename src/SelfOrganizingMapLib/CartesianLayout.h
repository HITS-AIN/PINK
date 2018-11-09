/**
 * @file   SelfOrganizingMapLib/CartesianLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstddef>
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
