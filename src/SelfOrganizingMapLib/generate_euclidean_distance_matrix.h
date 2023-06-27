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
#include "ImageProcessingLib/circular_euclidean_distance.h"
#include "UtilitiesLib/InputData.h"

namespace pink {

template <typename DataLayout, typename T>
void generate_euclidean_distance_matrix(std::vector<T>& euclidean_distance_matrix,
    std::vector<uint32_t>& best_rotation_matrix, uint32_t som_size, T const *som,
    DataLayout const& data_layout, uint32_t num_rot, std::vector<T> const& rotated_images,
    uint32_t euclidean_distance_dim, EuclideanDistanceShape const& euclidean_distance_shape)
{
    std::function<T(T const*, T const*, DataLayout const&, uint32_t)> ed_func;
    switch (euclidean_distance_shape)
    {
        case EuclideanDistanceShape::QUADRATIC:
        {
            ed_func = EuclideanDistanceFunctor<DataLayout>();
            break;
        }
        case EuclideanDistanceShape::CIRCULAR:
        {
            ed_func = CircularEuclideanDistanceFunctor<DataLayout>();
            break;
        }
    }

    for (uint32_t i = 0; i < som_size; ++i)
    {
        euclidean_distance_matrix[i] = ed_func(&som[i * data_layout.size()],
                   &rotated_images[0], data_layout, euclidean_distance_dim);

        #pragma omp parallel for
        for (uint32_t j = 1; j < num_rot; ++j)
        {
            auto tmp = ed_func(&som[i * data_layout.size()],
                       &rotated_images[j * data_layout.size()], data_layout, euclidean_distance_dim);

            #pragma omp critical
            if (tmp < euclidean_distance_matrix[i])
            {
                euclidean_distance_matrix[i] = tmp;
                best_rotation_matrix[i] = j;
            }
        }
    }
}

} // namespace pink
