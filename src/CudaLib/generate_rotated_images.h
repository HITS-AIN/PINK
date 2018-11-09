/**
 * @file   CudaLib/generate_rotated_images.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

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
template <typename LayoutType, typename T>
void generate_rotated_images(thrust::device_vector<T>& d_rotated_images, Data<LayoutType, T> const& data,
    uint32_t num_rot, uint32_t image_dim, uint32_t neuron_dim, bool useFlip, Interpolation interpolation,
    thrust::device_vector<T> const& d_cosAlpha, thrust::device_vector<T> const& d_sinAlpha)
{
    const uint16_t block_size = 32;
    uint32_t neuron_size = neuron_dim * neuron_dim;
    uint32_t image_size = image_dim * image_dim;

    auto&& d_image = data.get_device_vector();

    uint32_t spacing = data.get_layout().dimensionality > 2 ? data.get_dimension()[2] : 1;
    for (uint32_t i = 3; i != data.get_layout().dimensionality; ++i) spacing *= data.get_dimension()[i];

    // Crop first image
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_dim/block_size);
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid(gridSize, gridSize);

        // Start kernel
        for (uint32_t c = 0; c < spacing; ++c)
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
            for (uint32_t c = 0; c < spacing; ++c)
            {
                if (interpolation == Interpolation::NEAREST_NEIGHBOR)
                    rotate_and_crop_nearest_neighbor<<<dimGrid, dimBlock>>>(&d_rotated_images[(c + spacing) * neuron_size],
                        &d_image[c * image_size], neuron_size, neuron_dim, image_dim, &d_cosAlpha[0], &d_sinAlpha[0], spacing);
                else if (interpolation == Interpolation::BILINEAR)
                    rotate_and_crop_bilinear<<<dimGrid, dimBlock>>>(&d_rotated_images[(c + spacing) * neuron_size],
                        &d_image[c * image_size], neuron_size, neuron_dim, image_dim, &d_cosAlpha[0], &d_sinAlpha[0], spacing);
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

        int offset = num_rot/4 * spacing * neuron_size;
        int mc_neuron_size = spacing * neuron_size;

        // Start kernel
        for (uint32_t c = 0; c < spacing; ++c)
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
        dim3 dimGrid(gridSize, gridSize, num_rot * spacing);

        // Start kernel
        for (uint32_t c = 0; c < spacing; ++c)
        {
            flip<<<dimGrid, dimBlock>>>(&d_rotated_images[num_rot * spacing * neuron_size],
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
