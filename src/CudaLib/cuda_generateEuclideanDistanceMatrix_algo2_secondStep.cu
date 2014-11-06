/**
 * @file   cuda_generateEuclideanDistanceMatrix_algo2_secondStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cublas_v2.h"
#include <stdio.h>

#define BLOCK_SIZE 64

/**
 * CUDA Kernel Device code
 *
 * Reduce temp. array d_tmp to final arrays d_euclideanDistanceMatrix and d_bestRotationMatrix.
 */
template <unsigned int block_size>
__global__ void
second_step_kernel(float *euclideanDistanceMatrix, int *bestRotationMatrix, float *firstStep, int num_rot)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float *pFirstStep = firstStep + i*num_rot;
    float *pDist = euclideanDistanceMatrix + i;
    *pDist = pFirstStep[0];
	bestRotationMatrix[i] = 0;

    for (int n=1; n < num_rot; ++n) {
        if (pFirstStep[n] < *pDist) {
        	*pDist = pFirstStep[n];
        	bestRotationMatrix[i] = n;
        }
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix_algo2_secondStep(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix,
    float* d_firstStep, int som_size, int num_rot)
{
    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil((float)som_size/BLOCK_SIZE));

    // Start kernel
    second_step_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_euclideanDistanceMatrix, d_bestRotationMatrix, d_firstStep, num_rot);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_algo2_secondStep (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}
