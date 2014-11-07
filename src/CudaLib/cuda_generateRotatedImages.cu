/**
 * @file   cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
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
cuda_generateRotatedImages_kernel(float *rotatedImages, float *image, int image_size, int neuron_dim)
{
//	int tid = threadIdx.x;
//    int i = threadIdx.x;
//
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	float tx = float(x)*costheta - float(y)*sintheta;
//	float ty = float(x)*sintheta + float(y)*costheta;
//
//	float *pCurRot = rotatedImages + x * image_size;
//
//	if(x < neuron_dim && y < neuron_dim) {
//		pCurRot[x*width+y] = tex2D(image_texture, tx+0.5f, ty+0.5f);
//	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateRotatedImages(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim, bool flip)
{
	int image_size = image_dim * image_dim;

    // Setup execution parameters
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(num_rot);

    // Start kernel
    cuda_generateRotatedImages_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages, d_image, image_size, neuron_dim);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateRotatedImages (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
