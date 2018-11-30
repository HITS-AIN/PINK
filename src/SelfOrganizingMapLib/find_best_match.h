/**
 * @file   SelfOrganizingMapLib/find_best_match.h
 * @date   Oct 26, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdint>
#include <vector>

namespace pink {

template <typename T>
uint32_t find_best_match(std::vector<T> const& euclidean_distance_matrix, uint32_t som_size)
{
    uint32_t best_match = 0;
    T min_distance = euclidean_distance_matrix[0];
    for (uint32_t i = 1; i < som_size; ++i) {
        if (euclidean_distance_matrix[i] < min_distance) {
            min_distance = euclidean_distance_matrix[i];
            best_match = i;
        }
    }
    return best_match;
}

} // namespace pink
