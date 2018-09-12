/**
 * @file   UtilitiesLib/SOMInitialization.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

//! Type for SOM initialization
enum class SOMInitialization {
    ZERO,
    RANDOM,
    RANDOM_WITH_PREFERRED_DIRECTION,
    FILEINIT
};

//! Pretty printing of SOM layout type
inline std::ostream& operator << (std::ostream& os, SOMInitialization init)
{
    if (init == SOMInitialization::ZERO) os << "zero";
    else if (init == SOMInitialization::RANDOM) os << "random";
    else if (init == SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION) os << "random_with_preferred_direction";
    else if (init == SOMInitialization::FILEINIT) os << "file_init";
    else os << "undefined";
    return os;
}

} // namespace pink
