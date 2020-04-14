/**
 * @file   CudaLib/generate_euclidean_distance_matrix.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdio>
#include <thrust/device_vector.h>

#include "copy_and_transform_kernel.h"
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
template <typename NeuronLayout, typename T>
void generate_euclidean_distance_matrix(thrust::device_vector<T>& d_euclidean_distance_matrix,
    thrust::device_vector<uint32_t>& d_best_rotation_matrix, uint32_t som_size, NeuronLayout const& neuron_layout,
    thrust::device_vector<T> const& d_som, uint32_t number_of_spatial_transformations,
    thrust::device_vector<T> const& d_spatial_transformed_images, uint32_t block_size,
    DataType euclidean_distance_type, uint32_t euclidean_distance_dim,
    EuclideanDistanceShape const& euclidean_distance_shape,
    thrust::device_vector<uint32_t> const& d_circle_offset,
    thrust::device_vector<uint32_t> const& d_circle_delta)
{
    static thrust::device_vector<T> d_first_step(som_size * number_of_spatial_transformations);
    if (d_first_step.size() != som_size * number_of_spatial_transformations)
        d_first_step.resize(som_size * number_of_spatial_transformations);

    uint32_t euclidean_distance_size = 0;
    if (euclidean_distance_shape == EuclideanDistanceShape::QUADRATIC) {
        euclidean_distance_size = euclidean_distance_dim * euclidean_distance_dim * neuron_layout.get_spacing();
    } else if (euclidean_distance_shape == EuclideanDistanceShape::CIRCULAR) {
        euclidean_distance_size = d_circle_offset[euclidean_distance_dim] * neuron_layout.get_spacing();
    }

    auto d_som_size = som_size * euclidean_distance_size;
    auto d_spatial_transformed_images_size = number_of_spatial_transformations * euclidean_distance_size;
    auto neuron_dim = neuron_layout.get_last_dimension();
    auto offset = static_cast<uint32_t>((neuron_dim - euclidean_distance_dim) * 0.5);

    // First step ...
    if (euclidean_distance_type == DataType::UINT8)
    {
        static thrust::device_vector<uint8_t> d_som_uint8(d_som_size);
        if (d_som_uint8.size() != d_som_size)
            d_som_uint8.resize(d_som_size);

        static thrust::device_vector<uint8_t> d_spatial_transformed_images_uint8(d_spatial_transformed_images_size);
        if (d_spatial_transformed_images_uint8.size() != d_spatial_transformed_images_size)
            d_spatial_transformed_images_uint8.resize(d_spatial_transformed_images_size);

        // Setup execution parameters
        uint32_t grid_size = static_cast<uint32_t>(ceil(static_cast<float>(euclidean_distance_dim) / 16));

        dim3 dim_block(16, 16);
        dim3 dim_grid(grid_size, grid_size, som_size * neuron_layout.get_spacing());

        switch (euclidean_distance_shape)
        {
            case EuclideanDistanceShape::QUADRATIC:
            {
                copy_and_transform_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som_uint8[0]),
                    thrust::raw_pointer_cast(&d_som[0]), euclidean_distance_dim, neuron_dim, offset, 255);
                break;
            }
            case EuclideanDistanceShape::CIRCULAR:
            {
                copy_and_transform_circular_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som_uint8[0]),
                    thrust::raw_pointer_cast(&d_som[0]), euclidean_distance_dim, neuron_dim, offset, 255,
                    thrust::raw_pointer_cast(&d_circle_offset[0]), thrust::raw_pointer_cast(&d_circle_delta[0]));
                break;
            }
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        dim3 dim_grid2(grid_size, grid_size, number_of_spatial_transformations * neuron_layout.get_spacing());

        switch (euclidean_distance_shape)
        {
            case EuclideanDistanceShape::QUADRATIC:
            {
                copy_and_transform_kernel<<<dim_grid2, dim_block>>>(
                    thrust::raw_pointer_cast(&d_spatial_transformed_images_uint8[0]),
                    thrust::raw_pointer_cast(&d_spatial_transformed_images[0]),
                    euclidean_distance_dim, neuron_dim, offset, 255);
                break;
            }
            case EuclideanDistanceShape::CIRCULAR:
            {
                copy_and_transform_circular_kernel<<<dim_grid2, dim_block>>>(
                    thrust::raw_pointer_cast(&d_spatial_transformed_images_uint8[0]),
                    thrust::raw_pointer_cast(&d_spatial_transformed_images[0]),
                    euclidean_distance_dim, neuron_dim, offset, 255,
                    thrust::raw_pointer_cast(&d_circle_offset[0]), thrust::raw_pointer_cast(&d_circle_delta[0]));
                break;
            }
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        if (cuda_get_gpu_ids().size() > 1) {
            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_uint8, d_spatial_transformed_images_uint8,
                d_first_step, number_of_spatial_transformations, som_size, euclidean_distance_size, block_size);
        } else {
            generate_euclidean_distance_matrix_first_step(d_som_uint8, d_spatial_transformed_images_uint8,
                d_first_step, number_of_spatial_transformations, som_size, euclidean_distance_size, block_size);
        }
    }
    else if (euclidean_distance_type == DataType::UINT16)
    {
        static thrust::device_vector<uint16_t> d_som_uint16(d_som_size);
        if (d_som_uint16.size() != d_som_size)
            d_som_uint16.resize(d_som_size);

        static thrust::device_vector<uint16_t> d_spatial_transformed_images_uint16(d_spatial_transformed_images_size);
        if (d_spatial_transformed_images_uint16.size() != d_spatial_transformed_images_size)
            d_spatial_transformed_images_uint16.resize(d_spatial_transformed_images_size);

        // Setup execution parameters
        uint32_t grid_size = static_cast<uint32_t>(ceil(static_cast<float>(euclidean_distance_dim) / 16));

        dim3 dim_block(16, 16);
        dim3 dim_grid(grid_size, grid_size, som_size * neuron_layout.get_spacing());

        switch (euclidean_distance_shape)
        {
            case EuclideanDistanceShape::QUADRATIC:
            {
                copy_and_transform_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som_uint16[0]),
                    thrust::raw_pointer_cast(&d_som[0]), euclidean_distance_dim, neuron_dim, offset, 65535);
                break;
            }
            case EuclideanDistanceShape::CIRCULAR:
            {
                copy_and_transform_circular_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som_uint16[0]),
                    thrust::raw_pointer_cast(&d_som[0]), euclidean_distance_dim, neuron_dim, offset, 65535,
                    thrust::raw_pointer_cast(&d_circle_offset[0]), thrust::raw_pointer_cast(&d_circle_delta[0]));
                break;
            }
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        dim3 dim_grid2(grid_size, grid_size, number_of_spatial_transformations * neuron_layout.get_spacing());

        switch (euclidean_distance_shape)
        {
            case EuclideanDistanceShape::QUADRATIC:
            {
                copy_and_transform_kernel<<<dim_grid2, dim_block>>>(
                    thrust::raw_pointer_cast(&d_spatial_transformed_images_uint16[0]),
                    thrust::raw_pointer_cast(&d_spatial_transformed_images[0]),
                    euclidean_distance_dim, neuron_dim, offset, 65535);
                break;
            }
            case EuclideanDistanceShape::CIRCULAR:
            {
                copy_and_transform_circular_kernel<<<dim_grid2, dim_block>>>(
                    thrust::raw_pointer_cast(&d_spatial_transformed_images_uint16[0]),
                    thrust::raw_pointer_cast(&d_spatial_transformed_images[0]),
                    euclidean_distance_dim, neuron_dim, offset, 65535,
                    thrust::raw_pointer_cast(&d_circle_offset[0]), thrust::raw_pointer_cast(&d_circle_delta[0]));
                break;
            }
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        if (cuda_get_gpu_ids().size() > 1) {
            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_uint16, d_spatial_transformed_images_uint16,
                d_first_step, number_of_spatial_transformations, som_size, euclidean_distance_size, block_size);
        } else {
            generate_euclidean_distance_matrix_first_step(d_som_uint16, d_spatial_transformed_images_uint16,
                d_first_step, number_of_spatial_transformations, som_size, euclidean_distance_size, block_size);
        }
    }
    else if (euclidean_distance_type == DataType::FLOAT)
    {
        static thrust::device_vector<float> d_som_float(d_som_size);
        if (d_som_float.size() != d_som_size)
            d_som_float.resize(d_som_size);

        static thrust::device_vector<float> d_spatial_transformed_images_float(d_spatial_transformed_images_size);
        if (d_spatial_transformed_images_float.size() != d_spatial_transformed_images_size)
            d_spatial_transformed_images_float.resize(d_spatial_transformed_images_size);

        // Setup execution parameters
        uint32_t grid_size = static_cast<uint32_t>(ceil(static_cast<float>(euclidean_distance_dim) / 16));

        dim3 dim_block(16, 16);
        dim3 dim_grid(grid_size, grid_size, som_size * neuron_layout.get_spacing());

        switch (euclidean_distance_shape)
        {
            case EuclideanDistanceShape::QUADRATIC:
            {
                copy_and_transform_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som_float[0]),
                    thrust::raw_pointer_cast(&d_som[0]), euclidean_distance_dim, neuron_dim, offset, 1);
                break;
            }
            case EuclideanDistanceShape::CIRCULAR:
            {
                copy_and_transform_circular_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som_float[0]),
                    thrust::raw_pointer_cast(&d_som[0]), euclidean_distance_dim, neuron_dim, offset, 1,
                    thrust::raw_pointer_cast(&d_circle_offset[0]), thrust::raw_pointer_cast(&d_circle_delta[0]));
                break;
            }
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        dim3 dim_grid2(grid_size, grid_size, number_of_spatial_transformations * neuron_layout.get_spacing());

        switch (euclidean_distance_shape)
        {
            case EuclideanDistanceShape::QUADRATIC:
            {
                copy_and_transform_kernel<<<dim_grid2, dim_block>>>(
                    thrust::raw_pointer_cast(&d_spatial_transformed_images_float[0]),
                    thrust::raw_pointer_cast(&d_spatial_transformed_images[0]),
                    euclidean_distance_dim, neuron_dim, offset, 1);
                break;
            }
            case EuclideanDistanceShape::CIRCULAR:
            {
                copy_and_transform_circular_kernel<<<dim_grid2, dim_block>>>(
                    thrust::raw_pointer_cast(&d_spatial_transformed_images_float[0]),
                    thrust::raw_pointer_cast(&d_spatial_transformed_images[0]),
                    euclidean_distance_dim, neuron_dim, offset, 1,
                    thrust::raw_pointer_cast(&d_circle_offset[0]), thrust::raw_pointer_cast(&d_circle_delta[0]));
                break;
            }
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        if (cuda_get_gpu_ids().size() > 1) {
            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_float, d_spatial_transformed_images_float,
                d_first_step, number_of_spatial_transformations, som_size, euclidean_distance_size, block_size);
        } else {
            generate_euclidean_distance_matrix_first_step(d_som_float, d_spatial_transformed_images_float,
                d_first_step, number_of_spatial_transformations, som_size, euclidean_distance_size, block_size);
        }
    }
    else
    {
        throw pink::exception("Unknown euclidean_distance_type");
    }

    // Second step ...
    generate_euclidean_distance_matrix_second_step(d_euclidean_distance_matrix,
        d_best_rotation_matrix, d_first_step, number_of_spatial_transformations, som_size);
}

} // namespace pink
