/**
 * @file   SelfOrganizingMapLib/Dimension.h
 * @date   Nov 9, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstddef>
#include <ostream>

namespace pink {

inline std::ostream& operator << (std::ostream& os, std::array<uint32_t, 2> const& d)
{
	for (uint8_t i = 0; i < 1; ++i) os << d[i] << "x";
	return os << d[1];
}

} // namespace pink
