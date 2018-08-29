/**
 * @file   CheckArrays.cpp
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CheckArrays.h"
#include <iostream>

namespace pink {

void checkArrayForNan(float* a, int length, std::string const& msg)
{
    for (int i = 0; i < length; ++i) {
        if (a[i] != a[i]) {
            std::cout << msg << ": entry is nan." << std::endl;
            exit(1);
        }
    }
}

void checkArrayForNanAndNegative(float* a, int length, std::string const& msg)
{
    for (int i = 0; i < length; ++i) {
        if (a[i] != a[i]) {
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
