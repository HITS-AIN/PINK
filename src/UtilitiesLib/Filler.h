/**
 * @file   UtilitiesLib/Filler.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstddef>
#include <random>
#include <type_traits>

namespace pink {

/// Fill array with random numbers
template <class T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
void fill_random_uniform(T *a, std::size_t length, std::uint32_t seed = std::mt19937::default_seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0.0);

    for (std::size_t i = 0; i < length; ++i) {
        a[i] = dist(rng);
    }
}

/// Fill array with random numbers
template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void fill_random_uniform(T *a, std::size_t length, std::uint32_t seed = std::mt19937::default_seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(0);

    for (std::size_t i = 0; i < length; ++i) {
        a[i] = dist(rng);
    }
}

/// Fill array with a single value
template <class T>
void fill_value(T *a, std::size_t length, T value = 0)
{
    for (std::size_t i = 0; i < length; ++i) {
        a[i] = value;
    }
}

} // namespace pink
