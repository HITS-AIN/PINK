/**
 * @file   cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "crop_kernel.h"
#include "flip_kernel.h"
#include "rotateAndCrop_kernel.h"
#include "cublas_v2.h"
#include <stdio.h>

#define BLOCK_SIZE 32

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateRotatedImages(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim,
    bool useFlip, float *d_cosAlpha, float *d_sinAlpha)
{
	int neuron_size = neuron_dim * neuron_dim;
	int image_size = image_dim * image_dim;

	{
		// Setup execution parameters
		int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(gridSize, gridSize);

	    // Start kernel
		crop_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages, d_image, neuron_dim, image_dim);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch CUDA kernel crop_kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	{
		// Setup execution parameters
		int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(gridSize, gridSize, num_rot-1);

		// Start kernel
		rotateAndCrop_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + neuron_size, d_image,
			neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch CUDA kernel rotateAndCrop_kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}

	if (useFlip)
	{
		float *d_flippedImage = d_image + image_size;
		float *d_flippedRotatedImages = d_rotatedImages + num_rot * neuron_size;

		{
			// Setup execution parameters
			int gridSize = ceil((float)image_dim/BLOCK_SIZE);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid(gridSize, gridSize);

			// Start kernel
			flip_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_flippedImage, d_image, image_dim);

			cudaError_t error = cudaGetLastError();

			if (error != cudaSuccess)
			{
				fprintf(stderr, "Failed to launch CUDA kernel flip_kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		{
			// Setup execution parameters
			int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid(gridSize, gridSize);

			// Start kernel
			crop_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_flippedRotatedImages, d_flippedImage, neuron_dim, image_dim);

			cudaError_t error = cudaGetLastError();

			if (error != cudaSuccess)
			{
				fprintf(stderr, "Failed to launch CUDA kernel crop_kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		{
			// Setup execution parameters
			int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid(gridSize, gridSize, num_rot-1);

			// Start kernel
			rotateAndCrop_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_flippedRotatedImages + neuron_size, d_flippedImage,
				neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha);

			cudaError_t error = cudaGetLastError();

			if (error != cudaSuccess)
			{
				fprintf(stderr, "Failed to launch CUDA kernel rotateAndCrop_kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
	}
}
