/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep_opt.cu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdio.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
__global__ void euclidean_distance_opt_kernel(float *som, float *rotatedImages, float *firstStep, int neuron_size)
{
    float sum = 0.0;
    float *psom = som + blockIdx.x * neuron_size;
    float *prot = rotatedImages + threadIdx.x * neuron_size;

    for (int i = 0; i < neuron_size; ++i) {
        sum += __powf(psom[i] - prot[i], 2);
    }

    firstStep[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size)
{
    // Setup execution parameters
    dim3 dimBlock(num_rot);
    dim3 dimGrid(som_size);

    // Start kernel
    //printf("Starting CUDA Kernel with (%i,%i,%i) blocks and (%i,%i,%i) threads ...\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    euclidean_distance_opt_kernel<<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_firstStep_opt (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
