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
 * Computes the euclidean norm of array a and b.
 */
__global__ void
kernel(float *a, float *b, float *c, int length)
{
	int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float a_local[BLOCK_SIZE];
    __shared__ float b_local[BLOCK_SIZE];
    __shared__ float c_local[BLOCK_SIZE];

    a_local[tid] = (i < length) ? a[i] : 0.0;
    b_local[tid] = (i < length) ? b[i] : 0.0;
    c_local[tid] = 0.0;

    float tmp = a_local[tid] - b_local[tid];
    c_local[tid] += tmp * tmp;
    __syncthreads();

    // parallel reduction
    for (int s=1; s < blockDim.x; s *= 2) {
    	if (tid % (2*s) == 0) {
    	    c_local[tid] += c_local[tid + s];
    	}
    	__syncthreads();
    }

    if (tid == 0) c[blockIdx.x] = c_local[0];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
float cuda_calculateEuclideanDistanceWithoutSquareRoot(float *a, float *b, int length)
{
	float c = 0.0;
    unsigned int sizeInBytes = length * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b;

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

    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil((float)length/BLOCK_SIZE));

    // Array for intermediate sum
    float *d_isum;
    error = cudaMalloc((void **) &d_isum, dimGrid.x * sizeof(float));

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_isum returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Start kernel
    //printf("Starting CUDA Kernel with (%i,%i,%i) blocks and (%i,%i,%i) threads ...\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_isum, length);

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

    cublasStatus_t ret;
    cublasHandle_t handle;
    ret = cublasCreate(&handle);
    ret = cublasSasum(handle, dimGrid.x, d_isum, 1, &c);

    error = cudaFree(d_isum);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_isum (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return c;
}
