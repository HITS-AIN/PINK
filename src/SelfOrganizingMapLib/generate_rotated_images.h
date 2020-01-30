/**
 * @file   SelfOrganizingMapLib/generate_rotated_images.h
 * @date   Jan 30, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

#include "Data.h"
#include "ImageProcessingLib/crop.h"
#include "ImageProcessingLib/flip.h"
#include "ImageProcessingLib/resize.h"
#include "ImageProcessingLib/rotate.h"
#include "ImageProcessingLib/rotate_and_crop.h"
#include "ImageProcessingLib/rotate_90_degrees.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/// Primary template for SpatialTransformer
template <typename DataLayout>
struct SpatialTransformer
{
    template <typename NeuronLayout, typename T>
    auto operator () (Data<DataLayout, T> const& data, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, NeuronLayout const& neuron_layout) const;
};

/// SpatialTransformer: Specialization for CartesianLayout<1>
template <>
struct SpatialTransformer<CartesianLayout<1>>
{
    template <typename NeuronLayout, typename T>
    auto operator () ([[maybe_unused]] Data<CartesianLayout<1>, T> const& data,
        [[maybe_unused]] uint32_t number_of_rotations, [[maybe_unused]] bool use_flip,
        [[maybe_unused]] Interpolation interpolation, [[maybe_unused]] NeuronLayout const& neuron_layout) const
    {
        throw pink::exception("Not implemented yet.");
        return std::vector<T>();
    }
};

/// SpatialTransformer: Specialization for CartesianLayout<2>
template <>
struct SpatialTransformer<CartesianLayout<2>>
{
    template <typename NeuronLayout, typename T>
    auto operator () (Data<CartesianLayout<2>, T> const& data, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, NeuronLayout const& neuron_layout) const
    {
        // Images must be quadratic
        if (data.get_dimension()[0] != data.get_dimension()[1]) {
            throw pink::exception("Images must be quadratic.");
        }

        auto image_dim = data.get_dimension()[0];
        auto neuron_dim = neuron_layout.get_dimension()[0];
        auto neuron_size = neuron_dim * neuron_dim;

        uint32_t number_of_spatial_transformations = number_of_rotations * (use_flip ? 2 : 1);
        std::vector<T> rotated_images(number_of_spatial_transformations * neuron_size);

        uint32_t num_real_rot = number_of_rotations / 4;
        float angle_step_radians = static_cast<float>(2 * M_PI) / number_of_rotations;

        uint32_t offset1 = num_real_rot * neuron_size;
        uint32_t offset2 = 2 * offset1;
        uint32_t offset3 = 3 * offset1;

        // Copy original image to first position of image array
        T const *current_image = &data[0];
        T *current_rotated_image = &rotated_images[0];
        resize(current_image, current_rotated_image, image_dim, image_dim, neuron_dim, neuron_dim);
        if (number_of_rotations != 1) {
            rotate_90_degrees(current_rotated_image, current_rotated_image + offset1,
                neuron_dim, neuron_dim);
            rotate_90_degrees(current_rotated_image + offset1, current_rotated_image + offset2,
                neuron_dim, neuron_dim);
            rotate_90_degrees(current_rotated_image + offset2, current_rotated_image + offset3,
                neuron_dim, neuron_dim);
        }

        // Rotate images
        #pragma omp parallel for
        for (uint32_t i = 1; i < num_real_rot; ++i) {
            T const *current_image = &data[0];
            T *current_rotated_image = &rotated_images[i * neuron_size];
            rotate(current_image, current_rotated_image, image_dim, image_dim,
                neuron_dim, neuron_dim, i * angle_step_radians, interpolation);
            rotate_90_degrees(current_rotated_image, current_rotated_image + offset1,
                neuron_dim, neuron_dim);
            rotate_90_degrees(current_rotated_image + offset1, current_rotated_image + offset2,
                neuron_dim, neuron_dim);
            rotate_90_degrees(current_rotated_image + offset2, current_rotated_image + offset3,
                neuron_dim, neuron_dim);
        }

        // Flip images
        if (use_flip)
        {
            T *flipped_rotated_images = &rotated_images[number_of_rotations * neuron_size];

            #pragma omp parallel for
            for (uint32_t i = 0; i < number_of_rotations; ++i) {
                flip(&rotated_images[i * neuron_size], flipped_rotated_images + i * neuron_size,
                    neuron_dim, neuron_dim);
            }
        }

        return rotated_images;
    }
};

/// SpatialTransformer: Specialization for CartesianLayout<3>
template <>
struct SpatialTransformer<CartesianLayout<3>>
{
    template <typename NeuronLayout, typename T>
    auto operator () (Data<CartesianLayout<3>, T> const& data, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, NeuronLayout const& neuron_layout) const
    {
        // Images must be quadratic
        if (data.get_dimension()[1] != data.get_dimension()[2]) {
            throw pink::exception("Images must be quadratic.");
        }

        auto image_dim = data.get_dimension()[1];
        auto image_size = image_dim * image_dim;
        auto neuron_dim = neuron_layout.get_dimension()[1];
        auto neuron_size = neuron_dim * neuron_dim;
        auto spacing = data.get_dimension()[0];

        uint32_t number_of_spatial_transformations = number_of_rotations * (use_flip ? 2 : 1);
        std::vector<T> rotated_images(number_of_spatial_transformations * neuron_size * spacing);

        uint32_t num_real_rot = number_of_rotations / 4;
        float angle_step_radians = static_cast<float>(2 * M_PI) / number_of_rotations;

        uint32_t offset1 = num_real_rot * spacing * neuron_size;
        uint32_t offset2 = 2 * offset1;
        uint32_t offset3 = 3 * offset1;

        // Copy original image to first position of image array
        #pragma omp parallel for
        for (uint32_t i = 0; i < spacing; ++i)
        {
            T const *current_image = &data[i * image_size];
            T *current_rotated_image = &rotated_images[i * neuron_size];
            resize(current_image, current_rotated_image, image_dim, image_dim,
                neuron_dim, neuron_dim);
            if (number_of_rotations != 1) {
                rotate_90_degrees(current_rotated_image, current_rotated_image + offset1,
                    neuron_dim, neuron_dim);
                rotate_90_degrees(current_rotated_image + offset1, current_rotated_image + offset2,
                    neuron_dim, neuron_dim);
                rotate_90_degrees(current_rotated_image + offset2, current_rotated_image + offset3,
                    neuron_dim, neuron_dim);
            }
        }

        // Rotate images
        #pragma omp parallel for
        for (uint32_t i = 1; i < num_real_rot; ++i) {
            for (uint32_t j = 0; j < spacing; ++j) {
                T const *current_image = &data[j * image_size];
                T *current_rotated_image = &rotated_images[(i * spacing + j) * neuron_size];
                rotate(current_image, current_rotated_image, image_dim, image_dim,
                    neuron_dim, neuron_dim, i * angle_step_radians, interpolation);
                rotate_90_degrees(current_rotated_image, current_rotated_image + offset1,
                    neuron_dim, neuron_dim);
                rotate_90_degrees(current_rotated_image + offset1, current_rotated_image + offset2,
                    neuron_dim, neuron_dim);
                rotate_90_degrees(current_rotated_image + offset2, current_rotated_image + offset3,
                    neuron_dim, neuron_dim);
            }
        }

        // Flip images
        if (use_flip)
        {
            T *flipped_rotated_images = &rotated_images[spacing * number_of_rotations * neuron_size];

            #pragma omp parallel for
            for (uint32_t i = 0; i < number_of_rotations; ++i) {
                for (uint32_t j = 0; j < spacing; ++j) {
                    flip(&rotated_images[(i * spacing + j) * neuron_size],
                        flipped_rotated_images + (i * spacing + j) * neuron_size,
                        neuron_dim, neuron_dim);
                }
            }
        }

        return rotated_images;
    }
};

} // namespace pink
