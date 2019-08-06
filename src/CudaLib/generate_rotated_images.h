/**
 * @file   CudaLib/generate_rotated_images.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
//#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "resize_kernel.h"
#include "flip_kernel.h"
#include "rotate_90_degrees_list.h"
#include "rotate_bilinear_kernel.h"
#include "UtilitiesLib/Interpolation.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_rotated_images(thrust::device_vector<T>& d_rotated_images,
    thrust::device_vector<T> const& d_image, uint32_t spacing, uint32_t num_rot, uint32_t image_dim,
    uint32_t neuron_dim, bool useFlip, Interpolation interpolation,
    thrust::device_vector<float> const& d_cos_alpha, thrust::device_vector<float> const& d_sin_alpha)
{
    const uint32_t block_size = 32;
    const uint32_t neuron_size = neuron_dim * neuron_dim;
    const uint32_t image_size = image_dim * image_dim;

    thrust::fill(thrust::device, d_rotated_images.begin(), d_rotated_images.end(), 0.0);

    // Resize the first image
    {
        // Setup execution parameters
        auto min_dim = std::min(image_dim, neuron_dim);
        auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(min_dim) / block_size));
        dim3 dim_block(block_size, block_size);
        dim3 dim_grid(grid_size, grid_size);

        // Start kernel
        for (uint32_t c = 0; c < spacing; ++c)
        {
            resize_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_rotated_images[c * neuron_size]),
                thrust::raw_pointer_cast(&d_image[c * image_size]), neuron_dim, image_dim, min_dim);

            gpuErrchk(cudaPeekAtLastError());
        }
    }

    if (num_rot != 1)
    {
        assert(num_rot % 4 == 0);

        // Rotate images between 0 and 90 degrees
        {
            // Setup execution parameters
            auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
            auto num_real_rot = static_cast<uint32_t>(num_rot / 4) - 1;

            if (num_real_rot) {
                dim3 dim_block(block_size, block_size);
                dim3 dim_grid(grid_size, grid_size, num_real_rot);

                // Start kernel
                for (uint32_t c = 0; c < spacing; ++c)
                {
                    if (interpolation == Interpolation::BILINEAR) {
                        rotate_bilinear_kernel<<<dim_grid, dim_block>>>(
                            thrust::raw_pointer_cast(&d_image[c * image_size]),
                            thrust::raw_pointer_cast(&d_rotated_images[(c + spacing) * neuron_size]),
                            image_dim, image_dim, neuron_dim, neuron_dim,
                            thrust::raw_pointer_cast(&d_cos_alpha[0]),
                            thrust::raw_pointer_cast(&d_sin_alpha[0]), spacing);
                    } else {
                        throw pink::exception("generate_rotated_images: unknown interpolation type");
                    }

                    gpuErrchk(cudaPeekAtLastError());
                }
            }
        }

        // Special 90 degree rotation for remaining rotations between 90 and 360 degrees
        {
            // Setup execution parameters
            auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
            dim3 dim_block(block_size, block_size);
            dim3 dim_grid(grid_size, grid_size, num_rot/4);

            uint32_t offset = num_rot/4 * spacing * neuron_size;
            uint32_t mc_neuron_size = spacing * neuron_size;

            // Start kernel
            for (uint32_t c = 0; c < spacing; ++c)
            {
                rotate_90_degrees_list<<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(&d_rotated_images[c * neuron_size]),
                    neuron_dim, mc_neuron_size, offset);
                rotate_90_degrees_list<<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(&d_rotated_images[c * neuron_size + offset]),
                    neuron_dim, mc_neuron_size, offset);
                rotate_90_degrees_list<<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(&d_rotated_images[c * neuron_size + 2 * offset]),
                    neuron_dim, mc_neuron_size, offset);

                gpuErrchk(cudaPeekAtLastError());
            }
        }
    }

    if (useFlip)
    {
        // Setup execution parameters
        auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
        dim3 dim_block(block_size, block_size);
        dim3 dim_grid(grid_size, grid_size, num_rot * spacing);

        // Start kernel
        flip_kernel<<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(&d_rotated_images[num_rot * spacing * neuron_size]),
            thrust::raw_pointer_cast(&d_rotated_images[0]), neuron_dim, neuron_size);

        gpuErrchk(cudaPeekAtLastError());
    }
}

} // namespace pink
