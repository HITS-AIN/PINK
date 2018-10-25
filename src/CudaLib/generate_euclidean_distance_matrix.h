/**
 * @file   CudaLib/generate_euclidean_distance_matrix.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdio.h>
#include <thrust/device_vector.h>

#include "generate_euclidean_distance_matrix_first_step.h"
#include "generate_euclidean_distance_matrix_first_step_multi_gpu.h"
#include "generate_euclidean_distance_matrix_second_step.h"

namespace pink {

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_euclidean_distance_matrix(thrust::device_vector<T>& d_euclideanDistanceMatrix,
    thrust::device_vector<uint32_t>& d_bestRotationMatrix, uint32_t, number_of_neurons,
	thrust::device_vector<T> const& d_som, uint32_t num_rot,
	thrust::device_vector<T> const& d_rotatedImages, uint16_t block_size,
    bool useMultipleGPUs)
{
	thrust::device_vector<T> d_firstStep(som_size * num_rot);

    // First step ...
    if (useMultipleGPUs and cuda_getNumberOfGPUs() > 1) {
    	generate_euclidean_distance_matrix_first_step_multi_gpu(d_som, d_rotatedImages,
            d_firstStep, num_rot, block_size);
    } else {
    	generate_euclidean_distance_matrix_first_step(d_som, d_rotatedImages,
            d_firstStep, num_rot, block_size);
    }

    // Second step ...
    generate_euclidean_distance_matrix_second_step(d_euclideanDistanceMatrix,
        d_bestRotationMatrix, d_firstStep, num_rot);
}

} // namespace pink
