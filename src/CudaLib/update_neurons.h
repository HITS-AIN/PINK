/**
 * @file   CudaLib/update_neurons.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "CudaLib.h"

namespace pink {

/// Find the position where the euclidean distance is minimal between image and neuron.
template <typename T>
__global__
void find_best_matching_neuron_kernel(T const *euclidean_distance_matrix,
    uint32_t *best_match, uint32_t som_size)
{
    *best_match = 0;
    float min_distance = euclidean_distance_matrix[0];
    for (uint32_t i = 1; i < som_size; ++i) {
        if (euclidean_distance_matrix[i] < min_distance) {
            min_distance = euclidean_distance_matrix[i];
            *best_match = i;
        }
    }
}

/// CUDA Kernel Device code updating quadratic self organizing map using gaussian function.
template <unsigned int block_size, typename T>
__global__
void update_neurons_kernel(T *som, T const *rotated_images, uint32_t const *best_rotation_matrix,
    uint32_t best_match, float const *update_factors, uint32_t som_size, uint32_t neuron_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= neuron_size) return;

    float factor = update_factors[best_match * som_size + blockIdx.y];
    int pos = blockIdx.y * neuron_size + i;

    if (factor != 0.0)
    {
        som[pos] -= (som[pos] - rotated_images[best_rotation_matrix[blockIdx.y] * neuron_size + i]) * factor;
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void update_neurons(thrust::device_vector<T>& d_som, thrust::device_vector<T> const& d_rotated_images,
    thrust::device_vector<uint32_t> const& d_best_rotation_matrix,
    thrust::device_vector<T> const& d_euclidean_distance_matrix,
    thrust::device_vector<uint32_t>& d_best_match,
    thrust::device_vector<float> const& d_update_factors,
    uint32_t som_size, uint32_t neuron_size)
{
    {
        // Start kernel
        find_best_matching_neuron_kernel<<<1,1>>>(thrust::raw_pointer_cast(&d_euclidean_distance_matrix[0]),
            thrust::raw_pointer_cast(&d_best_match[0]), som_size);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    {
        // Setup execution parameters
        const uint32_t block_size = 32;
        const uint32_t grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_size) / block_size));
        dim3 dim_block(block_size);
        dim3 dim_grid(grid_size, som_size);

        // Start kernel
        update_neurons_kernel<block_size><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
            thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_best_rotation_matrix[0]),
            d_best_match[0], thrust::raw_pointer_cast(&d_update_factors[0]), som_size, neuron_size);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}

} // namespace pink
