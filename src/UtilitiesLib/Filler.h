/**
 * @file   UtilitiesLib/Filler.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <random>

/**
 * @brief Fill array with random numbers.
 */
template <class T>
void fillWithRandomNumbers(T *a, int length, int seed = 1234)
{
    typedef std::mt19937 MyRNG;
    MyRNG rng(seed);
    std::normal_distribution<T> normal_dist(0.0, 0.1);

    for (int i = 0; i < length; ++i) {
        a[i] = normal_dist(rng);
    }
}

/**
 * @brief Fill array with value.
 */
template <class T>
void fillWithValue(T *a, int length, T value = 0)
{
    for (int i = 0; i < length; ++i) {
        a[i] = value;
    }
}
