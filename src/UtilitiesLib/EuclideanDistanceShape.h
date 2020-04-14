/**
 * @file   UtilitiesLib/EuclideanDistanceShape.h
 * @date   Mar 11, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

/// Type for storage of intermediate SOMs
enum class EuclideanDistanceShape
{
    QUADRATIC,
    CIRCULAR
};

/// Pretty printing of EuclideanDistanceShape.
inline std::ostream& operator << (std::ostream& os, EuclideanDistanceShape type)
{
    if (type == EuclideanDistanceShape::QUADRATIC) os << "quadratic";
    else if (type == EuclideanDistanceShape::CIRCULAR) os << "circular";
    else os << "undefined";
    return os;
}

} // namespace pink
