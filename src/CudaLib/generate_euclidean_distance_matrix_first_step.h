/**
 * @file   CudaLib/generate_euclidean_distance_matrix_first_step.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <thrust/device_vector.h>

namespace pink {

/**
 * CUDA Kernel Device code
 *
 * Static loop unrolling for the thread within one warp.
 */
template <uint16_t block_size, typename T>
__device__
void warp_reduce(volatile T *data, int tid)
{
    if (block_size >= 64) data[tid] += data[tid + 32];
    if (block_size >= 32) data[tid] += data[tid + 16];
    if (block_size >= 16) data[tid] += data[tid +  8];
    if (block_size >=  8) data[tid] += data[tid +  4];
    if (block_size >=  4) data[tid] += data[tid +  2];
    if (block_size >=  2) data[tid] += data[tid +  1];
}

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <uint16_t block_size, typename T>
__global__
void euclidean_distance_kernel(T const *som, T const *rotated_images, T *firstStep, uint32_t neuron_size)
{
    int tid = threadIdx.x;
    T diff;
    T sum = 0.0;
    T const *psom = som + blockIdx.y * neuron_size;
    T const *prot = rotated_images + blockIdx.x * neuron_size;

    __shared__ T firstStep_local[block_size];

    for (uint32_t i = tid; i < neuron_size; i += block_size)
    {
        diff = psom[i] - prot[i];
        sum += diff * diff;
    }

    firstStep_local[tid] = sum;
    __syncthreads();

    // Parallel reduction
    if (block_size >= 512) { if (tid < 256) { firstStep_local[tid] += firstStep_local[tid + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) { firstStep_local[tid] += firstStep_local[tid + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tid <  64) { firstStep_local[tid] += firstStep_local[tid +  64]; } __syncthreads(); }

    // Static loop unrolling for the thread within one warp.
    if (tid < 32) warp_reduce<block_size>(firstStep_local, tid);

    // Copy accumulated local value to global array firstStep
    if (tid == 0) firstStep[blockIdx.x + blockIdx.y * gridDim.x] = firstStep_local[0];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_euclidean_distance_matrix_first_step(thrust::device_vector<T> const& d_som,
    thrust::device_vector<T> const& d_rotated_images, thrust::device_vector<T>& d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size, uint32_t neuron_size, uint16_t block_size)
{
    // Setup execution parameters
    dim3 dim_block(block_size);
    dim3 dim_grid(number_of_spatial_transformations, som_size);

    // Start kernel
    switch (block_size)
    {
        case 1024: euclidean_distance_kernel<1024><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case  512: euclidean_distance_kernel< 512><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case  256: euclidean_distance_kernel< 256><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case  128: euclidean_distance_kernel< 128><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case   64: euclidean_distance_kernel<  64><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case   32: euclidean_distance_kernel<  32><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        case   16: euclidean_distance_kernel<  16><<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_som[0]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), thrust::raw_pointer_cast(&d_first_step[0]), neuron_size); break;
        default:
        {
            fprintf(stderr, "generate_euclidean_distance_matrix_first_step: block size (%i) not supported.", block_size);
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
