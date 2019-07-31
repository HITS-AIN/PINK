/**
 * @file   UtilitiesLib/ipow.h
 * @date   Jul 31, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

/// Fill array with random numbers
template <typename T>
constexpr T ipow(T num, unsigned int pow)
{
    return (pow >= sizeof(unsigned int)*8) ? 0 :
        pow == 0 ? 1 : num * ipow(num, pow-1);
}

} // namespace pink
