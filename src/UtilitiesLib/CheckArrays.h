/**
 * @file   CheckArrays.h
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <string>

namespace pink {

void checkArrayForNan(float* a, int length, std::string const& msg);

void checkArrayForNanAndNegative(float* a, int length, std::string const& msg);

} // namespace pink
