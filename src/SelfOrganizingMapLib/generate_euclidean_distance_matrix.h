/**
 * @file   SelfOrganizingMapLib/generate_euclidean_distance_matrix.h
 * @date   Oct 26, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "ImageProcessingLib/euclidean_distance.h"

namespace pink {

template <typename T>
void generate_euclidean_distance_matrix(std::vector<T>& euclidean_distance_matrix,
    std::vector<uint32_t>& best_rotation_matrix, uint32_t som_size, T const *som,
    uint32_t image_size, uint32_t num_rot, std::vector<T> const& rotated_images)
{
    T tmp;
    T* pdist = &euclidean_distance_matrix[0];
    uint32_t* prot = &best_rotation_matrix[0];

    for (uint32_t i = 0; i < som_size; ++i) euclidean_distance_matrix[i] = std::numeric_limits<T>::max();

    for (uint32_t i = 0; i < som_size; ++i, ++pdist, ++prot) {
        #pragma omp parallel for private(tmp)
        for (uint32_t j = 0; j < num_rot; ++j) {
            tmp = euclidean_distance_square(&som[i * image_size], &rotated_images[j * image_size], image_size);
            #pragma omp critical
            if (tmp < *pdist) {
                *pdist = tmp;
                *prot = j;
            }
        }
    }
}

} // namespace pink
