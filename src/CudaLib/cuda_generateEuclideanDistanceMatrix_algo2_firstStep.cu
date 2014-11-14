/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cublas_v2.h"
#include <stdio.h>

#define BLOCK_SIZE 32

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void
euclidean_distance_kernel(float *som, float *rotatedImages, float *firstStep, int image_size)
{
	int tid = threadIdx.x;
    int i = threadIdx.x;
    float diff;

	__shared__ float firstStep_local[block_size];
	firstStep_local[tid] = 0.0f;

	__syncthreads();

    while(i < image_size)
    {
    	diff = som[i + blockIdx.y * image_size] - rotatedImages[i + blockIdx.x * image_size];
    	firstStep_local[tid] += diff * diff;
    	i += block_size;
    	__syncthreads();
    }

	// Parallel reduction
    if (block_size >= 512) { if (tid < 256) { firstStep_local[tid] += firstStep_local[tid + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) { firstStep_local[tid] += firstStep_local[tid + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tid <  64) { firstStep_local[tid] += firstStep_local[tid +  64]; } __syncthreads(); }

	if (tid < 32)
	{
		if (block_size >= 64) { firstStep_local[tid] += firstStep_local[tid + 32]; __syncthreads(); }
		if (block_size >= 32) { firstStep_local[tid] += firstStep_local[tid + 16]; __syncthreads(); }
		if (block_size >= 16) { firstStep_local[tid] += firstStep_local[tid +  8]; __syncthreads(); }
		if (block_size >=  8) { firstStep_local[tid] += firstStep_local[tid +  4]; __syncthreads(); }
		if (block_size >=  4) { firstStep_local[tid] += firstStep_local[tid +  2]; __syncthreads(); }
		if (block_size >=  2) { firstStep_local[tid] += firstStep_local[tid +  1]; __syncthreads(); }
	}

	// Copy accumulated local value to global array firstStep
	if (tid == 0) atomicExch(firstStep + blockIdx.x + blockIdx.y * gridDim.x, firstStep_local[tid]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix_algo2_firstStep(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int image_size)
{
    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(num_rot, som_size);

    // Start kernel
    euclidean_distance_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, image_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_algo2_firstStep (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
