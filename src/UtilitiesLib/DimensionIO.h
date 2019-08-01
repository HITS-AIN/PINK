/**
 * @file   SelfOrganizingMapLib/DimensionIO.h
 * @date   Nov 9, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstddef>
#include <ostream>

namespace pink {

/// Pretty print of dimension, 1-dim
std::ostream& operator << (std::ostream& os, std::array<uint32_t, 1> const& d);

/// Pretty print of dimension, 2-dim
std::ostream& operator << (std::ostream& os, std::array<uint32_t, 2> const& d);

/// Pretty print of dimension, 3-dim
std::ostream& operator << (std::ostream& os, std::array<uint32_t, 3> const& d);

} // namespace pink
