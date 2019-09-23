/**
 * @file   SelfOrganizingMapLib/generate_rotated_images.h
 * @date   Oct 26, 2018
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

/// If the input data is an image with two or more dimensions
/// it will be rotated in the plain spanned by the first two dimensions.
template <typename LayoutType, typename T>
auto generate_rotated_images(Data<LayoutType, T> const& data,
    uint32_t number_of_rotations, bool use_flip, Interpolation interpolation, uint32_t neuron_dim)
{
    // Images must have at least two dimensions
    if (data.get_layout().dimensionality < 2) {
        throw pink::exception("Date must have at least two dimensions for image rotation.");
    }
    // Images must be quadratic
    if (data.get_dimension()[0] != data.get_dimension()[1]) {
        throw pink::exception("Images must be quadratic.");
    }

    auto image_dim = data.get_dimension()[0];
    auto image_size = data.get_dimension()[0] * data.get_dimension()[1];
    auto neuron_size = neuron_dim * neuron_dim;

    uint32_t number_of_spatial_transformations = number_of_rotations * (use_flip ? 2 : 1);
    std::vector<T> rotated_images(number_of_spatial_transformations * neuron_size);

    uint32_t num_real_rot = number_of_rotations / 4;
    float angle_step_radians = static_cast<float>(2 * M_PI) / number_of_rotations;

    uint32_t spacing = data.get_layout().dimensionality > 2 ? data.get_dimension()[2] : 1;
    for (uint32_t i = 3; i < data.get_layout().dimensionality; ++i) spacing *= data.get_dimension()[i];

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

} // namespace pink
