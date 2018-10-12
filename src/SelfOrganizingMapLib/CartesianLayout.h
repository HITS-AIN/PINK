/**
 * @file   SelfOrganizingMapLib/CartesianLayout.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stddef.h>
#include <array>
#include <cstdint>

namespace pink {

template <size_t dim>
struct CartesianLayout
{
	static const size_t dimensionality = dim;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;
};

} // namespace pink
