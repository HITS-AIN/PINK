/**
 * @file   CheckArrays.h
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <string>

namespace pink {

void check_array_for_nan(float* a, int length, std::string const& msg);

void check_array_for_nan_and_negative(float* a, int length, std::string const& msg);

} // namespace pink
