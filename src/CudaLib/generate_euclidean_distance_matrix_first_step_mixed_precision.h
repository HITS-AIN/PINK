/**
 * @file   CudaLib/generate_euclidean_distance_matrix_first_step_mixed_precision.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "euclidean_distance_kernel.h"

namespace pink {

/// Host function that prepares data array and passes it to the CUDA kernel
template <typename DataType, typename EuclideanType>
void generate_euclidean_distance_matrix_first_step_mixed_precision(thrust::device_vector<EuclideanType> const& d_som,
    thrust::device_vector<EuclideanType> const& d_rotated_images, thrust::device_vector<DataType>& d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size, uint32_t neuron_size, uint16_t block_size)
{
    // Setup execution parameters
    dim3 dim_block(block_size);
    dim3 dim_grid(number_of_spatial_transformations, som_size);

    // Start kernel
    switch (block_size)
    {
//        case 1024: euclidean_distance_kernel<1024><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
//                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
//        case  512: euclidean_distance_kernel< 512><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
//                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case  256: euclidean_distance_kernel< 256><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
//        case  128: euclidean_distance_kernel< 128><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
//                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
//        case   64: euclidean_distance_kernel<  64><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
//                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
//        case   32: euclidean_distance_kernel<  32><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
//                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
//        case   16: euclidean_distance_kernel<  16><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
//                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        default:
        {
            fprintf(stderr, "generate_euclidean_distance_matrix_first_step_mixed_precision: block size (%i) not supported.", block_size);
            exit(EXIT_FAILURE);
        }
    }

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel euclidean_distance_kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

} // namespace pink
