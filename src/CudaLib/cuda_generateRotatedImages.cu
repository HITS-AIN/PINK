/**
 * @file   cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "crop_kernel.h"
#include "flip_kernel.h"
#include "rotateAndCropNearestNeighbor_kernel.h"
#include "rotateAndCropBilinear_kernel.h"
#include "rotate90degreesList_kernel.h"
#include <stdio.h>

namespace pink {

#define BLOCK_SIZE 32

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateRotatedImages(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim,
    bool useFlip, Interpolation interpolation, float *d_cosAlpha, float *d_sinAlpha, int numberOfChannels)
{
    int neuron_size = neuron_dim * neuron_dim;
    int image_size = image_dim * image_dim;

    // Crop first image
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(gridSize, gridSize);

        // Start kernel
        for (int c = 0; c < numberOfChannels; ++c)
        {
            crop_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size,
                d_image + c*image_size, neuron_dim, image_dim);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel crop_kernel (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }

    if (num_rot == 1) return;

    // Rotate images between 0 and 90 degrees
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
        int num_real_rot = num_rot/4-1;

        if (num_real_rot) {
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid(gridSize, gridSize, num_real_rot);

            // Start kernel
            for (int c = 0; c < numberOfChannels; ++c)
            {
                if (interpolation == NEAREST_NEIGHBOR)
                    rotateAndCropNearestNeighbor_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + (c+numberOfChannels)*neuron_size, d_image + c*image_size,
                        neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha, numberOfChannels);
                else if (interpolation == BILINEAR)
                    rotateAndCropBilinear_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + (c+numberOfChannels)*neuron_size, d_image + c*image_size,
                        neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha, numberOfChannels);
                else {
                    fprintf(stderr, "cuda_generateRotatedImages: unknown interpolation type!\n");
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
    }

    // Special 90 degree rotation for remaining rotations between 90 and 360 degrees
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(gridSize, gridSize, num_rot/4);

        int offset = num_rot/4 * numberOfChannels * neuron_size;
        int mc_neuron_size = numberOfChannels * neuron_size;

        // Start kernel
        for (int c = 0; c < numberOfChannels; ++c)
        {
            rotate90degreesList_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size,
                neuron_dim, mc_neuron_size, offset);
            rotate90degreesList_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size + offset,
                neuron_dim, mc_neuron_size, offset);
            rotate90degreesList_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size + 2*offset,
                neuron_dim, mc_neuron_size, offset);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel rotate90degrees_kernel (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }

    if (useFlip)
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(gridSize, gridSize, num_rot * numberOfChannels);

        // Start kernel
        for (int c = 0; c < numberOfChannels; ++c)
        {
            flip_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + num_rot * numberOfChannels * neuron_size,
                d_rotatedImages, neuron_dim, neuron_size);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel flip_kernel (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }
}

} // namespace pink
