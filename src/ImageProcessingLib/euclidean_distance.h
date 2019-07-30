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
T euclidean_distance_square(T const *a, T const *b, uint32_t length)
{
    std::vector<T> diff(length);
    for (uint32_t i = 0; i < length; ++i) diff[i] = a[i] - b[i];
    return dot(diff);
}

/// Same as @euclidean_distance but without square root for speed (sum((a[i] - b[i])^2))
template <typename T>
T euclidean_distance_square_offset(T const *a, T const *b, uint32_t image_dim,
    uint32_t euclidean_distance_dim)
{
    uint32_t offset = static_cast<uint32_t>((image_dim - euclidean_distance_dim) * 0.5);
    std::vector<T> diff(euclidean_distance_dim * euclidean_distance_dim);
    for (uint32_t i = 0; i < euclidean_distance_dim; ++i)
      for (uint32_t j = 0; j < euclidean_distance_dim; ++j)
        diff[i * euclidean_distance_dim + j] = a[(i + offset) * image_dim + j + offset]
                                             - b[(i + offset) * image_dim + j + offset];
    return dot(diff);
}

/// Returns euclidean distance of two arrays (sqrt(sum((a[i] - b[i])^2))
template <typename T>
T euclidean_distance(T const *a, T const *b, uint32_t length)
{
    return std::sqrt(euclidean_distance_square(a, b, length));
}

} // namespace pink
