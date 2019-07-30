/**
 * @file   CheckArrays.cpp
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <iostream>

#include "CheckArrays.h"

namespace pink {

void check_array_for_nan(float* a, int length, std::string const& msg)
{
    for (int i = 0; i < length; ++i) {
        if (std::isnan(a[i])) {
            std::cout << msg << ": entry is nan." << std::endl;
            exit(1);
        }
    }
}

void check_array_for_nan_and_negative(float* a, int length, std::string const& msg)
{
    for (int i = 0; i < length; ++i) {
        if (std::isnan(a[i])) {
            std::cout << msg << ": entry is nan." << std::endl;
            exit(1);
        }
        if (a[i] < 0.0) {
            std::cout << msg << ": entry is < 0." << std::endl;
            exit(1);
        }
    }
}

} // namespace pink
