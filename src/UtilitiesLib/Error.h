/**
 * @file   UtilitiesLib/Error.h
 * @date   Nov 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

/**
 * @brief Exit with fatal error message
 */
inline void fatalError(std::string const& msg)
{
    std::cout << "FATAL ERROR: " << msg << std::endl;
    exit(1);
}

} // namespace pink
