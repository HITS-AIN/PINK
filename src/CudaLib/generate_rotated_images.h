/**
 * @file   CudaLib/generate_rotated_images.h
 * @date   Feb 4, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
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

/// Primary template for SpatialTransformer (GPU)
template <typename DataLayout>
struct SpatialTransformerGPU
{
    template <typename NeuronLayout, typename T>
    void operator () (thrust::device_vector<T>& d_rotated_images,
        thrust::device_vector<T> const& d_image, uint32_t number_of_rotations,
        bool use_flip, Interpolation interpolation,
        DataLayout const& data_layout,
        NeuronLayout const& neuron_layout,
        thrust::device_vector<float> const& d_cos_alpha,
        thrust::device_vector<float> const& d_sin_alpha) const;
};

/// SpatialTransformer (GPU): Specialization for CartesianLayout<1>
template <>
struct SpatialTransformerGPU<CartesianLayout<1>>
{
    template <typename NeuronLayout, typename T>
    void operator () ([[maybe_unused]] thrust::device_vector<T>& d_rotated_images,
        [[maybe_unused]] thrust::device_vector<T> const& d_image,
        [[maybe_unused]] uint32_t number_of_rotations,
        [[maybe_unused]] bool use_flip, [[maybe_unused]] Interpolation interpolation,
        [[maybe_unused]] CartesianLayout<1> const& data_layout,
        [[maybe_unused]] NeuronLayout const& neuron_layout,
        [[maybe_unused]] thrust::device_vector<float> const& d_cos_alpha,
        [[maybe_unused]] thrust::device_vector<float> const& d_sin_alpha) const
    {
        throw pink::exception("Not implemented yet.");
    }
};

/// SpatialTransformer (GPU): Specialization for CartesianLayout<2>
template <>
struct SpatialTransformerGPU<CartesianLayout<2>>
{
    template <typename NeuronLayout, typename T>
    void operator () (thrust::device_vector<T>& d_rotated_images,
        thrust::device_vector<T> const& d_image, uint32_t number_of_rotations,
        bool use_flip, Interpolation interpolation,
        CartesianLayout<2> const& data_layout,
        NeuronLayout const& neuron_layout,
        thrust::device_vector<float> const& d_cos_alpha,
        thrust::device_vector<float> const& d_sin_alpha) const
    {
        // Images must be quadratic
        if (data_layout.get_dimension()[0] != data_layout.get_dimension()[1]) {
            throw pink::exception("Images must be quadratic.");
        }

        const uint32_t block_size = 32;

        auto image_dim = data_layout.get_dimension()[0];
        auto neuron_dim = neuron_layout.get_dimension()[0];
        auto neuron_size = neuron_dim * neuron_dim;

        uint32_t number_of_spatial_transformations = number_of_rotations * (use_flip ? 2 : 1);
        std::vector<T> rotated_images(number_of_spatial_transformations * neuron_size);

        thrust::fill(thrust::device, d_rotated_images.begin(), d_rotated_images.end(), 0.0);

        // Resize the first image
        {
            // Setup execution parameters
            auto min_dim = std::min(image_dim, neuron_dim);
            auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(min_dim) / block_size));
            dim3 dim_block(block_size, block_size);
            dim3 dim_grid(grid_size, grid_size);

            resize_kernel<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(&d_rotated_images[0]),
                thrust::raw_pointer_cast(&d_image[0]), neuron_dim, image_dim, min_dim);

            gpuErrchk(cudaPeekAtLastError());
        }

        if (number_of_rotations != 1)
        {
            assert(number_of_rotations % 4 == 0);

            // Rotate images between 0 and 90 degrees
            {
                // Setup execution parameters
                auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
                auto num_real_rot = static_cast<uint32_t>(number_of_rotations / 4) - 1;

                if (num_real_rot) {
                    dim3 dim_block(block_size, block_size);
                    dim3 dim_grid(grid_size, grid_size, num_real_rot);

                    if (interpolation == Interpolation::BILINEAR) {
                        rotate_bilinear_kernel<<<dim_grid, dim_block>>>(
                            thrust::raw_pointer_cast(&d_image[0]),
                            thrust::raw_pointer_cast(&d_rotated_images[neuron_size]),
                            image_dim, image_dim, neuron_dim, neuron_dim,
                            thrust::raw_pointer_cast(&d_cos_alpha[0]),
                            thrust::raw_pointer_cast(&d_sin_alpha[0]), 1);
                    } else {
                        throw pink::exception("generate_rotated_images: unknown interpolation type");
                    }

                    gpuErrchk(cudaPeekAtLastError());
                }
            }

            // Special 90 degree rotation for remaining rotations between 90 and 360 degrees
            {
                // Setup execution parameters
                auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
                dim3 dim_block(block_size, block_size);
                dim3 dim_grid(grid_size, grid_size, number_of_rotations/4);

                uint32_t offset = number_of_rotations/4 * neuron_size;

				rotate_90_degrees_list<<<dim_grid, dim_block>>>(
					thrust::raw_pointer_cast(&d_rotated_images[0]),
					neuron_dim, neuron_size, offset);
				gpuErrchk(cudaPeekAtLastError());

				rotate_90_degrees_list<<<dim_grid, dim_block>>>(
					thrust::raw_pointer_cast(&d_rotated_images[offset]),
					neuron_dim, neuron_size, offset);
				gpuErrchk(cudaPeekAtLastError());

				rotate_90_degrees_list<<<dim_grid, dim_block>>>(
					thrust::raw_pointer_cast(&d_rotated_images[2 * offset]),
					neuron_dim, neuron_size, offset);
				gpuErrchk(cudaPeekAtLastError());
            }
        }

        if (use_flip)
        {
            // Setup execution parameters
            auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
            dim3 dim_block(block_size, block_size);
            dim3 dim_grid(grid_size, grid_size, number_of_rotations);

            flip_kernel<<<dim_grid, dim_block>>>(
                thrust::raw_pointer_cast(&d_rotated_images[number_of_rotations * neuron_size]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), neuron_dim, neuron_size);

            gpuErrchk(cudaPeekAtLastError());
        }
    }
};

/// SpatialTransformer (GPU): Specialization for CartesianLayout<3>
template <>
struct SpatialTransformerGPU<CartesianLayout<3>>
{
    template <typename NeuronLayout, typename T>
    auto operator () (thrust::device_vector<T>& d_rotated_images,
        thrust::device_vector<T> const& d_image, uint32_t number_of_rotations,
        bool use_flip, Interpolation interpolation,
        CartesianLayout<3> const& data_layout,
        NeuronLayout const& neuron_layout,
        thrust::device_vector<float> const& d_cos_alpha,
        thrust::device_vector<float> const& d_sin_alpha) const
    {
        // Images must be quadratic
        if (data_layout.get_dimension()[1] != data_layout.get_dimension()[2]) {
            throw pink::exception("Images must be quadratic.");
        }

        const uint32_t block_size = 32;

        auto image_dim = data_layout.get_dimension()[1];
        auto image_size = image_dim * image_dim;
        auto neuron_dim = neuron_layout.get_dimension()[1];
        auto neuron_size = neuron_dim * neuron_dim;
        auto spacing = data_layout.get_dimension()[0];

        uint32_t number_of_spatial_transformations = number_of_rotations * (use_flip ? 2 : 1);
        std::vector<T> rotated_images(number_of_spatial_transformations * neuron_size * spacing);

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

        if (number_of_rotations != 1)
        {
            assert(number_of_rotations % 4 == 0);

            // Rotate images between 0 and 90 degrees
            {
                // Setup execution parameters
                auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
                auto num_real_rot = static_cast<uint32_t>(number_of_rotations / 4) - 1;

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
                dim3 dim_grid(grid_size, grid_size, number_of_rotations/4);

                uint32_t offset = number_of_rotations/4 * spacing * neuron_size;
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

        if (use_flip)
        {
            // Setup execution parameters
            auto grid_size = static_cast<uint32_t>(ceil(static_cast<float>(neuron_dim) / block_size));
            dim3 dim_block(block_size, block_size);
            dim3 dim_grid(grid_size, grid_size, number_of_rotations * spacing);

            // Start kernel
            flip_kernel<<<dim_grid, dim_block>>>(
                thrust::raw_pointer_cast(&d_rotated_images[number_of_rotations * spacing * neuron_size]),
                thrust::raw_pointer_cast(&d_rotated_images[0]), neuron_dim, neuron_size);

            gpuErrchk(cudaPeekAtLastError());
        }
    }
};

} // namespace pink
