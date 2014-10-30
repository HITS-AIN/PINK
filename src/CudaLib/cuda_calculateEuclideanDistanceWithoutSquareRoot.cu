/**
 * @file   cuda_calculateEuclideanDistanceWithoutSquareRoot.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include <stdio.h>

#define BLOCK_SIZE 32

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean norm of array a and b.
 */
__global__ void
kernel(float *a, float *b, float *c, int length)
{
	int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float a_local[BLOCK_SIZE];
    __shared__ float b_local[BLOCK_SIZE];

    a_local[tid] = (i < length) ? a[i] : 0.0;
    b_local[tid] = (i < length) ? b[i] : 0.0;
    __syncthreads();

    float tmp;
    tmp = a_local[i] - b_local[i];
    *c += tmp * tmp;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
float cuda_calculateEuclideanDistanceWithoutSquareRoot(float *a, float *b, int length)
{
	float c = 0.0;
    unsigned int sizeInBytes = length * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b, *d_c;

    cudaError_t error;

    error = cudaMalloc((void **) &d_a, sizeInBytes);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_a returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_b, sizeInBytes);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_b returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_c, sizeof(float));

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_c returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_a, a, sizeInBytes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_a returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_b, b, sizeInBytes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_b returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_c, &c, sizeof(float), cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_b returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(length/BLOCK_SIZE);

    printf("Starting CUDA Kernel with (%i,%i,%i) blocks and (%i,%i,%i) threads ...\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);

    kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, length);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Free device global memory
    error = cudaFree(d_a);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_a (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_b);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_b (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return c;
}
