/**
 * @file   cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "crop_kernel.h"
#include "flip_kernel.h"
#include "rotateAndCrop_kernel.h"
#include "rotateAndCropBilinear_kernel.h"
#include "rotate90degreesList_kernel.h"
#include <stdio.h>

#define BLOCK_SIZE 32

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateRotatedImages(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim,
    bool useFlip, Interpolation interpolation, float *d_cosAlpha, float *d_sinAlpha)
{
	int neuron_size = neuron_dim * neuron_dim;

	// Crop first image
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

	// Rotate images between 0 and 90 degrees
	{
		// Setup execution parameters
		int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
		int num_real_rot = num_rot/4-1;

		if (num_real_rot) {
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid(gridSize, gridSize, num_real_rot);

			// Start kernel
			if (interpolation == NEAREST_NEIGHBOR)
			    rotateAndCrop_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + neuron_size, d_image,
				    neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha);
			else if (interpolation == BILINEAR)
			    rotateAndCropBilinear_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + neuron_size, d_image,
				    neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha);
			else {
				fprintf(stderr, "cuda_generateRotatedImages: unkown interpolation type!\n");
				exit(EXIT_FAILURE);
			}

			cudaError_t error = cudaGetLastError();

			if (error != cudaSuccess)
			{
				fprintf(stderr, "Failed to launch CUDA kernel rotateAndCrop_kernel (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
	}

	// Special 90 degree rotation for remaining rotations between 90 and 360 degrees
	{
		// Setup execution parameters
		int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(gridSize, gridSize, num_rot/4);

		// Start kernel
		rotate90degreesList_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages, neuron_dim, neuron_size, num_rot/4*neuron_size);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch CUDA kernel rotate90degrees_kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}

	if (useFlip)
	{
		// Setup execution parameters
		int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(gridSize, gridSize, num_rot);

		// Start kernel
		flip_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + num_rot * neuron_size, d_rotatedImages, neuron_dim, neuron_size);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch CUDA kernel flip_kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
}
