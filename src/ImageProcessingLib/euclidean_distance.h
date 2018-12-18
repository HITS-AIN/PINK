/**
 * @file   ImageProcessingLib/euclidean_distance.h
 * @date   Dec 18, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

/// Returns dot product of array with itself
template <typename T>
T dot(std::vector<T> const& v)
{
    T dot = 0;
    for (auto&& e : v) dot += e * e;
    return dot;
}

/// Same as @euclidean_distance but without square root for speed (sum((a[i] - b[i])^2))
template <typename T>
T euclidean_distance_square(T const *a, T const *b, int length)
{
    std::vector<T> diff(length);
    for (int i = 0; i < length; ++i) diff[i] = a[i] - b[i];
    return dot(diff);
}

/// Returns euclidean distance of two arrays (sqrt(sum((a[i] - b[i])^2))
template <typename T>
T euclidean_distance(T const *a, T const *b, int length)
{
    return std::sqrt(euclidean_distance_square(a, b, length));
}

} // namespace pink
