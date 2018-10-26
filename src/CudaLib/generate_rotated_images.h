/**
 * @file   CudaLib/generate_rotated_images.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdio>
#include <thrust/device_vector.h>

#include "crop.h"
#include "flip.h"
#include "ImageProcessingLib/Interpolation.h"
#include "rotate_and_crop_nearest_neighbor.h"
#include "rotate_and_crop_bilinear.h"
#include "rotate_90degrees_list.h"

namespace pink {

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_rotated_images(thrust::device_vector<T>& d_rotated_images, thrust::device_vector<T> const& d_image,
    uint32_t num_rot, uint32_t image_dim, uint32_t neuron_dim, bool useFlip, Interpolation interpolation,
    thrust::device_vector<T> const& d_cosAlpha, thrust::device_vector<T> const& d_sinAlpha, uint32_t numberOfChannels)
{
    const uint8_t block_size = 32;
    uint32_t neuron_size = neuron_dim * neuron_dim;
    uint32_t image_size = image_dim * image_dim;

    // Crop first image
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/block_size);
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid(gridSize, gridSize);

        // Start kernel
        for (uint32_t c = 0; c < numberOfChannels; ++c)
        {
            crop<<<dimGrid, dimBlock>>>(&d_rotated_images[c * neuron_size],
                &d_image[c * image_size], neuron_dim, image_dim);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel crop (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }

    if (num_rot == 1) return;

    // Rotate images between 0 and 90 degrees
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/block_size);
        int num_real_rot = num_rot/4-1;

        if (num_real_rot) {
            dim3 dimBlock(block_size, block_size);
            dim3 dimGrid(gridSize, gridSize, num_real_rot);

            // Start kernel
            for (uint32_t c = 0; c < numberOfChannels; ++c)
            {
                if (interpolation == Interpolation::NEAREST_NEIGHBOR)
                    rotate_and_crop_nearest_neighbor<<<dimGrid, dimBlock>>>(&d_rotated_images[(c + numberOfChannels) * neuron_size],
                        &d_image[c * image_size], neuron_size, neuron_dim, image_dim, &d_cosAlpha[0], &d_sinAlpha[0], numberOfChannels);
                else if (interpolation == Interpolation::BILINEAR)
                    rotate_and_crop_bilinear<<<dimGrid, dimBlock>>>(&d_rotated_images[(c + numberOfChannels) * neuron_size],
                        &d_image[c * image_size], neuron_size, neuron_dim, image_dim, &d_cosAlpha[0], &d_sinAlpha[0], numberOfChannels);
                else {
                    fprintf(stderr, "generate_rotated_images_gpu: unknown interpolation type!\n");
                    exit(EXIT_FAILURE);
                }

                cudaError_t error = cudaGetLastError();

                if (error != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch CUDA kernel rotateAndCrop (error code %s)!\n", cudaGetErrorString(error));
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    // Special 90 degree rotation for remaining rotations between 90 and 360 degrees
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/block_size);
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid(gridSize, gridSize, num_rot/4);

        int offset = num_rot/4 * numberOfChannels * neuron_size;
        int mc_neuron_size = numberOfChannels * neuron_size;

        // Start kernel
        for (uint32_t c = 0; c < numberOfChannels; ++c)
        {
            rotate_90degrees_list<<<dimGrid, dimBlock>>>(&d_rotated_images[c * neuron_size],
                neuron_dim, mc_neuron_size, offset);
            rotate_90degrees_list<<<dimGrid, dimBlock>>>(&d_rotated_images[c * neuron_size + offset],
                neuron_dim, mc_neuron_size, offset);
            rotate_90degrees_list<<<dimGrid, dimBlock>>>(&d_rotated_images[c * neuron_size + 2 * offset],
                neuron_dim, mc_neuron_size, offset);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel rotate_90degrees_list (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }

    if (useFlip)
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/block_size);
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid(gridSize, gridSize, num_rot * numberOfChannels);

        // Start kernel
        for (uint32_t c = 0; c < numberOfChannels; ++c)
        {
            flip<<<dimGrid, dimBlock>>>(&d_rotated_images[num_rot * numberOfChannels * neuron_size],
                &d_rotated_images[0], neuron_dim, neuron_size);

            cudaError_t error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch CUDA kernel flip (error code %s)!\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }
}

} // namespace pink
