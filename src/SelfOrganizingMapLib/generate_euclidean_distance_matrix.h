/**
 * @file   SelfOrganizingMapLib/generate_euclidean_distance_matrix.h
 * @date   Jan 29, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "ImageProcessingLib/euclidean_distance.h"

namespace pink {

template <typename DataLayout, typename T>
void generate_euclidean_distance_matrix(std::vector<T>& euclidean_distance_matrix,
    std::vector<uint32_t>& best_rotation_matrix, uint32_t som_size, T const *som,
	DataLayout const& data_layout, uint32_t num_rot, std::vector<T> const& rotated_images,
	uint32_t euclidean_distance_dim)
{
    T tmp;
    T* pdist = &euclidean_distance_matrix[0];
    uint32_t* prot = &best_rotation_matrix[0];

    std::fill(euclidean_distance_matrix.begin(), euclidean_distance_matrix.end(), std::numeric_limits<T>::max());

    for (uint32_t i = 0; i < som_size; ++i, ++pdist, ++prot) {
        #pragma omp parallel for private(tmp)
        for (uint32_t j = 0; j < num_rot; ++j) {
            tmp = EuclideanDistanceFunctor<DataLayout>()(&som[i * data_layout.size()],
                &rotated_images[j * data_layout.size()], data_layout, euclidean_distance_dim);
            #pragma omp critical
            if (tmp < *pdist) {
                *pdist = tmp;
                *prot = j;
            }
        }
    }
}

} // namespace pink
