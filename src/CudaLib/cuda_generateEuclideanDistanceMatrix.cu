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
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void
euclidean_distance_kernel(float *a, float *b, float *c, int image_size)
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
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void
euclidean_distance_kernel2(float *a, float *b, float *c, int image_size)
{
	int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int som_idx = blockIdx.y;
    int rot_idx = blockIdx.z;

	__shared__ float c_local[block_size];

    if (i < image_size) {
		float diff1 = a[i + som_idx * image_size] - b[i + rot_idx * image_size];
		float diff2 = a[i + som_idx * image_size + block_size] - b[i + rot_idx * image_size + block_size];
		c_local[tid] = diff1 * diff1;
		c_local[tid + block_size] = diff2 * diff2;
    } else {
    	c_local[tid] = 0.0f;
    }

	__syncthreads();

	if (tid < 32) {
		if (block_size >= 64) c_local[tid] += c_local[tid + 32];
		if (block_size >= 32) c_local[tid] += c_local[tid + 16];
		if (block_size >= 16) c_local[tid] += c_local[tid +  8];
		if (block_size >=  8) c_local[tid] += c_local[tid +  4];
		if (block_size >=  4) c_local[tid] += c_local[tid +  2];
		if (block_size >=  2) c_local[tid] += c_local[tid +  1];
	}

	if (tid == 0) c[blockIdx.x*blockDim.y*blockDim.z + blockIdx.y*blockDim.z + blockIdx.z] = c_local[0];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, int som_dim, float* d_som,
    int image_dim, int num_rot, float* d_rotatedImages)
{
	unsigned int image_size = image_dim * image_dim;
	unsigned int som_size = som_dim * som_dim;
	unsigned int red_size = ((float)image_size/BLOCK_SIZE);

    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(red_size, som_size, num_rot);

    float *d_tmp = cuda_alloc_float(dimGrid.x * dimGrid.y * dimGrid.z);

    // Start kernel
    //printf("Starting CUDA Kernel with (%i,%i,%i) blocks and (%i,%i,%i) threads ...\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
    euclidean_distance_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_tmp, image_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Call second step for final reduction
    cuda_generateEuclideanDistanceMatrix_secondStep(d_euclideanDistanceMatrix, d_bestRotationMatrix, d_tmp,
        image_size, num_rot, red_size);

    cuda_free(d_tmp);
}
