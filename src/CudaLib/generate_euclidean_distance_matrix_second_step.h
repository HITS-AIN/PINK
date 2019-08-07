/**
 * @file   CudaLib/generate_euclidean_distance_matrix_second_step.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <thrust/device_vector.h>

namespace pink {

/**
 * CUDA Kernel Device code
 *
 * Reduce temp. array d_tmp to final arrays d_euclidean_distance_matrix and d_best_rotation_matrix.
 */
template <typename T>
__global__
void second_step_kernel(T *euclidean_distance_matrix, uint32_t *best_rotation_matrix, T const *first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= som_size) return;

    T const *pFirstStep = first_step + i * number_of_spatial_transformations;
    T *pDist = euclidean_distance_matrix + i;

    *pDist = pFirstStep[0];
    best_rotation_matrix[i] = 0;

    for (uint32_t n = 1; n < number_of_spatial_transformations; ++n) {
        if (pFirstStep[n] < *pDist) {
            *pDist = pFirstStep[n];
            best_rotation_matrix[i] = n;
        }
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_euclidean_distance_matrix_second_step(thrust::device_vector<T>& d_euclidean_distance_matrix,
    thrust::device_vector<uint32_t>& d_best_rotation_matrix, thrust::device_vector<T> const& d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size)
{
    const uint32_t block_size = 16;
    const uint32_t grid_size = static_cast<uint32_t>(ceil(static_cast<float>(som_size) / block_size));

    // Setup execution parameters
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    // Start kernel
    second_step_kernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(&d_euclidean_distance_matrix[0]),
        thrust::raw_pointer_cast(&d_best_rotation_matrix[0]), thrust::raw_pointer_cast(&d_first_step[0]),
        number_of_spatial_transformations, som_size);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace pink
