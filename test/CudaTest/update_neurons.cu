/**
 * @file   CudaTest/update_neurons.cu
 * @date   Nov 14, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "CudaLib/update_neurons.h"

using namespace pink;

TEST(UpdateNeuronsTest, update_neurons)
{
    std::vector<float> som(16, 0);
    std::vector<float> rotated_images{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint32_t> best_rotation_matrix{0, 1, 0, 0};
    std::vector<float> euclidean_distance_matrix{2, 1, 3, 4};
    std::vector<uint32_t> best_match(1);
    std::vector<float> update_factors{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    thrust::device_vector<float> d_som = som;
    thrust::device_vector<float> d_rotated_images = rotated_images;
    thrust::device_vector<uint32_t> d_best_rotation_matrix = best_rotation_matrix;
    thrust::device_vector<float> d_euclidean_distance_matrix = euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_match = best_match;
    thrust::device_vector<float> d_update_factors = update_factors;

    update_neurons(d_som, d_rotated_images,
        d_best_rotation_matrix, d_euclidean_distance_matrix, d_best_match, d_update_factors, 4, 4);

    thrust::host_vector<uint32_t> result_best_match = d_best_match;
    EXPECT_EQ(1UL, result_best_match[0]);

    thrust::host_vector<uint32_t> result = d_som;
    thrust::host_vector<uint32_t> expected = std::vector<uint32_t>{0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(expected, result);
}
