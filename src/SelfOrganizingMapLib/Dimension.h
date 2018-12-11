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

inline std::ostream& operator << (std::ostream& os, std::array<uint32_t, 1> const& d)
{
    return os << d[0];
}

inline std::ostream& operator << (std::ostream& os, std::array<uint32_t, 2> const& d)
{
    return os << d[0] << "x" << d[1];
}

inline std::ostream& operator << (std::ostream& os, std::array<uint32_t, 3> const& d)
{
    return os << d[0] << "x" << d[1] << "x" << d[2];
}

} // namespace pink
