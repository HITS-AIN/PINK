/**
 * @file   UtilitiesLib/get_file_header.h
 * @date   Jun 11, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>
#include <string>

namespace pink {

/// Returns header of binary data file
/// File stream position is set behind the header section
/// or unchanged if no header was found
std::string get_file_header(std::istream& ifs);

} // namespace pink
