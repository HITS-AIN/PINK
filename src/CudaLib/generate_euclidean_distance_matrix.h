/**
 * @file   CudaLib/generate_euclidean_distance_matrix.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "CudaLib.h"
#include "generate_euclidean_distance_matrix_first_step.h"
#include "generate_euclidean_distance_matrix_first_step_multi_gpu.h"
#include "generate_euclidean_distance_matrix_second_step.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_euclidean_distance_matrix(thrust::device_vector<T>& d_euclidean_distance_matrix,
    thrust::device_vector<uint32_t>& d_best_rotation_matrix, uint32_t som_size, uint32_t neuron_size,
    thrust::device_vector<T> const& d_som, uint32_t number_of_spatial_transformations,
    thrust::device_vector<T> const& d_spatial_transformed_images, uint16_t block_size,
    DataType euclidean_distance_type)
{
    static thrust::device_vector<T> d_first_step(som_size * number_of_spatial_transformations);
    if (d_first_step.size() != som_size * number_of_spatial_transformations)
    	d_first_step.resize(som_size * number_of_spatial_transformations);

    // First step ...
    if (euclidean_distance_type == DataType::UINT8)
    {
    	static thrust::device_vector<uint8_t> d_som_uint8(d_som.size());
    	if (d_som_uint8.size() != d_som.size())
    		d_som_uint8.resize(d_som.size());

    	static thrust::device_vector<uint8_t> d_spatial_transformed_images_uint8(d_spatial_transformed_images.size());
    	if (d_spatial_transformed_images_uint8.size() != d_spatial_transformed_images.size())
    		d_spatial_transformed_images_uint8.resize(d_spatial_transformed_images.size());

        thrust::transform(d_som.begin(), d_som.end(), d_som.begin(), d_som_uint8.begin(),
            [=] __host__ __device__ (T x, [[ maybe_unused ]] T y) {
            assert(x >= 0.0 and x <= 1.0);
            return x * 256;
        });

        thrust::transform(d_spatial_transformed_images.begin(), d_spatial_transformed_images.end(),
            d_spatial_transformed_images.begin(), d_spatial_transformed_images_uint8.begin(),
            [=] __host__ __device__ (T x, [[ maybe_unused ]] T y) {
            assert(x >= 0.0 and x <= 1.0);
            return x * 256;
        });

        if (cuda_get_gpu_ids().size() > 1) {
            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_uint8, d_spatial_transformed_images_uint8,
                d_first_step, number_of_spatial_transformations, som_size, neuron_size, block_size);
        } else {
            generate_euclidean_distance_matrix_first_step(d_som_uint8, d_spatial_transformed_images_uint8,
                d_first_step, number_of_spatial_transformations, som_size, neuron_size, block_size);
        }
    } else if (euclidean_distance_type == DataType::FLOAT) {
        if (cuda_get_gpu_ids().size() > 1) {
            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som, d_spatial_transformed_images,
                d_first_step, number_of_spatial_transformations, som_size, neuron_size, block_size);
        } else {
            generate_euclidean_distance_matrix_first_step(d_som, d_spatial_transformed_images,
                d_first_step, number_of_spatial_transformations, som_size, neuron_size, block_size);
        }
    } else {
        throw pink::exception("Unknown euclidean_distance_type");
    }

    // Second step ...
    generate_euclidean_distance_matrix_second_step(d_euclidean_distance_matrix,
        d_best_rotation_matrix, d_first_step, number_of_spatial_transformations, som_size);
}

} // namespace pink
