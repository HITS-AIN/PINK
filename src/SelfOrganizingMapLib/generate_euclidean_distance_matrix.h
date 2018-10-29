/**
 * @file   SelfOrganizingMapLib/generate_euclidean_distance_matrix.h
 * @date   Oct 26, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdint>
#include <limits>
#include <vector>

#include "ImageProcessingLib/ImageProcessing.h"

namespace pink {

template <typename T>
void generate_euclidean_distance_matrix(std::vector<T>& euclidean_distance_matrix,
    std::vector<uint32_t>& best_rotation_matrix, int som_size, std::vector<T> const& som,
    int image_size,	int num_rot, std::vector<T> const& rotated_images)
{
    T tmp;
    T* pdist = &euclidean_distance_matrix[0];
    uint32_t* prot = &best_rotation_matrix[0];
    T* psom = nullptr;

    for (int i = 0; i < som_size; ++i) euclidean_distance_matrix[i] = std::numeric_limits<T>::max();

    for (int i = 0; i < som_size; ++i, ++pdist, ++prot) {
        psom = &som[i * image_size];
        #pragma omp parallel for private(tmp)
        for (int j = 0; j < num_rot; ++j) {
            tmp = calculateEuclideanDistanceWithoutSquareRoot(psom, &rotated_images[j * image_size], image_size);
            #pragma omp critical
            if (tmp < *pdist) {
                *pdist = tmp;
                *prot = j;
            }
        }
    }
}

} // namespace pink
