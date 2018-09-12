/**
 * @file   UtilitiesLib/DistributionFunction.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

//! Type for distribution function for SOM update
enum class DistributionFunction {
    GAUSSIAN,
    MEXICANHAT
};

//! Pretty printing of distribution function
inline std::ostream& operator << (std::ostream& os, DistributionFunction function)
{
    if (function == DistributionFunction::GAUSSIAN) os << "gaussian";
    else if (function == DistributionFunction::MEXICANHAT) os << "mexicanhat";
    else os << "undefined";
    return os;
}

} // namespace pink
