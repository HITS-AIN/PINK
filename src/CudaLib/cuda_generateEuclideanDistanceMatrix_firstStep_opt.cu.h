/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep_opt.cu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "cublas_v2.h"
#include <stdio.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void euclidean_distance_opt_kernel(float *som, float *rotatedImages, float *diff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    diff[i] = som[i] - rotatedImages[i];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size)
{
    // Setup execution parameters
    int grid_size = ceil((float)neuron_size/block_size);
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    float *d_diff = cuda_alloc_float(neuron_size);

    // Start kernel
    for (int i = 0; i < num_rot; ++i)
    {
        for (int j = 0; j < som_size; ++j)
        {
            euclidean_distance_opt_kernel<block_size><<<dimGrid, dimBlock>>>(d_som + j * neuron_size,
                d_rotatedImages + i * neuron_size, d_diff);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_firstStep (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }

    cuda_free(d_diff);
}
