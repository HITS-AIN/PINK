/**
 * @file   ImageProcessing.h
 * @brief  Plain-C functions for image processing.
 * @date   Oct 7, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

/// Interpolation type for images.
enum class Interpolation
{
    NEAREST_NEIGHBOR,  ///< Refuse values behind the comma.
    BILINEAR           ///< Interpolate value by distance to pixels.
};

inline std::ostream& operator << (std::ostream& os, Interpolation interpolation)
{
    if (interpolation == Interpolation::NEAREST_NEIGHBOR) os << "nearest_neighbor";
    else if (interpolation == Interpolation::BILINEAR) os << "bilinear";
    else os << "undefined";
    return os;
}

} // namespace pink
