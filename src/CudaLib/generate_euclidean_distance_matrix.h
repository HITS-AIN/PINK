/**
 * @file   CudaLib/generate_euclidean_distance_matrix.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdio>
#include <thrust/device_vector.h>

#include "CudaLib.h"
#include "generate_euclidean_distance_matrix_first_step.h"
//#include "generate_euclidean_distance_matrix_first_step_multi_gpu.h"
#include "generate_euclidean_distance_matrix_second_step.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_euclidean_distance_matrix(thrust::device_vector<T>& d_euclidean_distance_matrix,
    thrust::device_vector<uint32_t>& d_best_rotation_matrix, uint32_t number_of_neurons, uint32_t neuron_size,
    thrust::device_vector<T> const& d_som, uint32_t number_of_spatial_transformations,
    thrust::device_vector<T> const& d_rotated_images, uint16_t block_size,
    bool use_multiple_gpus)
{
    thrust::device_vector<T> d_first_step(number_of_neurons * number_of_spatial_transformations);

    // First step ...
    if (use_multiple_gpus and cuda_getNumberOfGPUs() > 1) {
    	pink::exception("Multi GPUs are not supported.");
        //generate_euclidean_distance_matrix_first_step_multi_gpu(d_som, d_rotated_images,
        //    d_first_step, number_of_spatial_transformations, block_size);
    } else {
        generate_euclidean_distance_matrix_first_step(d_som, d_rotated_images,
            d_first_step, number_of_spatial_transformations, number_of_neurons, neuron_size, block_size);
    }

    // Second step ...
    generate_euclidean_distance_matrix_second_step(d_euclidean_distance_matrix,
        d_best_rotation_matrix, d_first_step, number_of_spatial_transformations, number_of_neurons);
}

} // namespace pink
