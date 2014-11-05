/**
 * @file   cuda_calculateEuclideanDistanceWithoutSquareRoot.cu
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
second_step_kernel(float *a, int *b, float *c, int image_size)
{
	int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int som_idx = blockIdx.y;
    int rot_idx = blockIdx.z;

    __shared__ float a_local[block_size];
    __shared__ float b_local[block_size];
    __shared__ float c_local[block_size];

    a_local[tid] = (i < image_size) ? a[i + som_idx * image_size] : 0.0;
    b_local[tid] = (i < image_size) ? b[i + rot_idx * image_size] : 0.0;

    float diff = a_local[tid] - b_local[tid];
    c_local[tid] = diff * diff;
    __syncthreads();

    // parallel reduction
    for (int s=1; s < blockDim.x; s *= 2) {
    	if (tid % (2*s) == 0) {
    	    c_local[tid] += c_local[tid + s];
    	}
    	__syncthreads();
    }

    if (tid == 0) c[blockIdx.x*blockDim.y*blockDim.z + blockIdx.y*blockDim.z + blockIdx.z] = c_local[0];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix_secondStep(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, float* d_tmp,
    int image_size, int num_rot, int red_size)
{
    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(1);

    // Start kernel
    second_step_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_euclideanDistanceMatrix, d_bestRotationMatrix, d_tmp, image_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}
