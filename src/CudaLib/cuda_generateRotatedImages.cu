/**
 * @file   cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cublas_v2.h"
#include <stdio.h>

#define BLOCK_SIZE 32

texture<float, 2, cudaReadModeElementType> image_texture;

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void
cuda_generateRotatedImages_kernel(float *rotatedImages, float *image, int neuron_size, int neuron_dim, float* cosAlpha, float *sinAlpha)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float cosAlpha_local = cosAlpha[blockIdx.z];
	float sinAlpha_local = sinAlpha[blockIdx.z];

	float tx = float(x)*cosAlpha_local - float(y)*sinAlpha_local;
	float ty = float(x)*sinAlpha_local + float(y)*cosAlpha_local;

	float *pCurRot = rotatedImages + blockIdx.z * neuron_size;

	pCurRot[x*neuron_dim + y] = tex2D(image_texture, tx+0.5f, ty+0.5f);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateRotatedImages(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim,
    bool flip, float *d_cosAlpha, float *d_sinAlpha)
{
	int image_size = image_dim * image_dim;
	int neuron_size = neuron_dim * neuron_dim;

	cudaBindTexture(0, image_texture, d_image, image_size * sizeof(float));

	// Copy original image on first position
	//crop(image_dim, image_dim, neuron_dim, neuron_dim, image, rotatedImages);

    // Setup execution parameters
	int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(gridSize, gridSize, num_rot);

    // Start kernel
    cuda_generateRotatedImages_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages, d_image,
        neuron_size, neuron_dim, d_cosAlpha, d_sinAlpha);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateRotatedImages (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
