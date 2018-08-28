/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep_opt3.cu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "cublas_v2.h"
#include <stdio.h>

namespace pink {

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void euclidean_distance_opt3_kernel(float *som, float *rotatedImages, float *diff, int neuron_size)
{
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j=0; j < neuron_size; ++j)
        diff[j] = som[j] - rotatedImages[j];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt3(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size)
{
    // Setup execution parameters
    int grid_size = ceil((float)neuron_size/block_size);
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size, som_size, num_rot);

    float *d_diff = cuda_alloc_float(som_size * num_rot * neuron_size);

    // Start kernel
    euclidean_distance_opt3_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_diff, neuron_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_firstStep_opt3 (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

//    cublasHandle_t handle;
//    cublasCreate(&handle);
//    cublasSdot(handle, dimGrid.x, d_isum, 1, &c);

    cuda_free(d_diff);
}

} // namespace pink
