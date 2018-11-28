/**
 * @file   UtilitiesLib/DataType.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <ostream>

#include "pink_exception.h"

namespace pink {

//! Type for execution path
enum class DataType {
	FLOAT,
    DOUBLE,
    HALF,
	UINT16,
    UINT8
};

//! Pretty printing of IntermediateStorageType.
inline std::ostream& operator << (std::ostream& os, DataType type)
{
    if (type == DataType::FLOAT) os << "float";
    else if (type == DataType::DOUBLE) os << "double";
    else if (type == DataType::HALF) os << "half";
    else if (type == DataType::UINT16) os << "uint16";
    else if (type == DataType::UINT8) os << "uint8";
    else pink::exception("Undefined DataType");
    return os;
}

} // namespace pink
