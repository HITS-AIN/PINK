/**
 * @file   SelfOrganizingMapLib/DimensionIO.cpp
 * @date   Nov 9, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "DimensionIO.h"

namespace pink {

std::ostream& operator << (std::ostream& os, std::array<uint32_t, 1> const& d)
{
    return os << d[0];
}

std::ostream& operator << (std::ostream& os, std::array<uint32_t, 2> const& d)
{
    return os << d[0] << " x " << d[1];
}

std::ostream& operator << (std::ostream& os, std::array<uint32_t, 3> const& d)
{
    return os << d[0] << " x " << d[1] << " x " << d[2];
}

} // namespace pink
