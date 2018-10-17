/**
 * @file   SelfOrganizingMapLib/HexagonalLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stddef.h>
#include <array>
#include <cstdint>
#include <numeric>

namespace pink {

struct HexagonalLayout
{
    static const size_t dimensionality = 1;
    static constexpr const char* type = "HexagonalLayout";

    typedef uint32_t IndexType;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;

    uint32_t get_size() const {
        return std::accumulate(dimension.begin(), dimension.end(), 1, std::multiplies<uint32_t>());
    }

    DimensionType dimension;
};

} // namespace pink
