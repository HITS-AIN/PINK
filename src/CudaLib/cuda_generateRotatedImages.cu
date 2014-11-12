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
 * Computes multiple rotations of an image. cosine and sin
 */
template <unsigned int block_size>
__global__ void
cuda_generateRotatedImages_kernel(float *rotatedImages, float *image, int neuron_size,
    int neuron_dim, int image_dim, float* cosAlpha, float *sinAlpha)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= neuron_dim or y >= neuron_dim) return;

	float cosAlpha_local = cosAlpha[blockIdx.z];
	float sinAlpha_local = sinAlpha[blockIdx.z];

	int x0 = neuron_dim * 0.5;
	int y0 = neuron_dim * 0.5;
	int tx = (x-x0)*cosAlpha_local + (y-y0)*sinAlpha_local + x0;
	int ty = (y-y0)*cosAlpha_local - (x-x0)*sinAlpha_local + y0;

	float *pCurRot = rotatedImages + blockIdx.z * neuron_size;

    //pCurRot[x*neuron_dim + y] = tex2D(image_texture, tx+0.5f, ty+0.5f);

    if (tx >= 0 and tx < image_dim and ty >= 0 and ty < image_dim) {
    	atomicAdd(pCurRot + x*neuron_dim + y, image[tx*image_dim + ty]);
    } else {
    	atomicAdd(pCurRot + x*neuron_dim + y, 0.0f);
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateRotatedImages(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim,
    bool flip, float *d_cosAlpha, float *d_sinAlpha)
{
	int neuron_size = neuron_dim * neuron_dim;

	cuda_fill_zero(d_rotatedImages, num_rot * neuron_size);

	//cudaBindTexture(0, image_texture, d_image, image_size * sizeof(float));

	// Copy original image on first position
	//crop(image_dim, image_dim, neuron_dim, neuron_dim, image, rotatedImages);

    // Setup execution parameters
	int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(gridSize, gridSize, num_rot);

    // Start kernel
    cuda_generateRotatedImages_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages, d_image,
        neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateRotatedImages (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
