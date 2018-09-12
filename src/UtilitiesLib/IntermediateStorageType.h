/**
 * @file   UtilitiesLib/IntermediateStorageType.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

//! Type for storage of intermediate SOMs
enum class IntermediateStorageType {
    OFF,
    OVERWRITE,
    KEEP
};

//! Pretty printing of IntermediateStorageType.
inline std::ostream& operator << (std::ostream& os, IntermediateStorageType type)
{
    if (type == IntermediateStorageType::OFF) os << "off";
    else if (type == IntermediateStorageType::OVERWRITE) os << "overwrite";
    else if (type == IntermediateStorageType::KEEP) os << "keep";
    else os << "undefined";
    return os;
}

} // namespace pink
