/**
 * @file   SelfOrganizingMapLib/HexagonalLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stddef.h>
#include <array>
#include <cstdint>

namespace pink {

struct HexagonalLayout
{
	static const size_t dimensionality = 1;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;
};

} // namespace pink
