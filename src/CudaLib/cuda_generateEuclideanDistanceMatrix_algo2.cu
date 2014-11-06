/**
 * @file   cuda_calculateEuclideanDistanceWithoutSquareRoot.cu
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
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float diff;

	__shared__ float firstStep_local[block_size];
	firstStep_local[tid] = 0.0f;

    while(i < image_size)
    {
    	diff = som[i + blockIdx.y * image_size] - rotatedImages[i + blockIdx.z * image_size];
    	firstStep_local[tid] += diff * diff;
    	i += block_size;
    	__syncthreads();
    }

	// Parallel reduction
	if (tid < 32)
	{
		if (block_size >= 64) { firstStep_local[tid] += firstStep_local[tid + 32]; __syncthreads(); }
		if (block_size >= 32) { firstStep_local[tid] += firstStep_local[tid + 16]; __syncthreads(); }
		if (block_size >= 16) { firstStep_local[tid] += firstStep_local[tid +  8]; __syncthreads(); }
		if (block_size >=  8) { firstStep_local[tid] += firstStep_local[tid +  4]; __syncthreads(); }
		if (block_size >=  4) { firstStep_local[tid] += firstStep_local[tid +  2]; __syncthreads(); }
		if (block_size >=  2) { firstStep_local[tid] += firstStep_local[tid +  1]; __syncthreads(); }
	}

	if (tid == 0) firstStep[blockIdx.y*blockDim.z + blockIdx.z] += firstStep_local[0];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix_algo2(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, int som_dim, float* d_som,
    int image_dim, int num_rot, float* d_rotatedImages)
{
	unsigned int image_size = image_dim * image_dim;
	unsigned int som_size = som_dim * som_dim;
	//unsigned int red_size = ceil((float)image_size/BLOCK_SIZE);
	unsigned int red_size = 1;

    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(red_size, som_size, num_rot);

    float *d_firstStep = cuda_alloc_float(som_size * num_rot);
    cuda_fill_zero(d_firstStep, som_size * num_rot);

    // Start kernel
    //printf("Starting CUDA Kernel with (%i,%i,%i) blocks and (%i,%i,%i) threads ...\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
    euclidean_distance_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, image_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_algo2 (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	float *firstStep = new float[som_size * num_rot];
	cuda_copyDeviceToHost_float(firstStep, d_firstStep, som_size * num_rot);

	for (int i=0; i < som_size; ++i)
		printf("gpu eucl %i: %f\n", i, firstStep[i*num_rot]);

    cuda_generateEuclideanDistanceMatrix_algo2_secondStep(d_euclideanDistanceMatrix, d_bestRotationMatrix,
        d_firstStep, som_size, num_rot);

    cuda_free(d_firstStep);
}
