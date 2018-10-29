/**
 * @file   SelfOrganizingMapLib/generate_rotated_images.h
 * @date   Oct 26, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdint>
#include <vector>

#include "Data.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

/// If the input data is an image with two or more dimensions
/// it will be rotated in the plain spanned by the first two dimensions.
template <typename LayoutType, typename T>
auto generate_rotated_images(Data<LayoutType, T> const& data,
    uint32_t num_rot, bool use_flip, Interpolation interpolation, uint32_t new_image_size)
{
    std::vector<T> rotated_images;

    // Images must be quadratic
    if (data.get_dimension()[0] != data.get_dimension()[1]) pink::exception("Images must be quadratic.");
    auto&& image_size = data.get_dimension()[0];

    int num_real_rot = num_rot/4;
    T angleStepRadians = 2.0 * M_PI / num_rot;

    uint32_t spacing = 0;
	if (data.dimensionality == 3) spacing = data.get_dimension()[2];
	for (uint32_t i = 3; i != data.dimensionality == 3; ++i) spacing *= data.get_dimension()[i];

    int offset1 = num_real_rot * spacing * neuron_size;
    int offset2 = 2 * offset1;
    int offset3 = 3 * offset1;

    // Copy original image to first position of image array
    #pragma omp parallel for
    for (int c = 0; c < spacing; ++c) {
        T *currentImage = image + c*image_size;
        T *currentrotated_images = rotated_images + c*neuron_size;
        crop(image_dim, image_dim, neuron_dim, neuron_dim, currentImage, currentrotated_images);
        rotate_90degrees(neuron_dim, neuron_dim, currentrotated_images, currentrotated_images + offset1);
        rotate_90degrees(neuron_dim, neuron_dim, currentrotated_images + offset1, currentrotated_images + offset2);
        rotate_90degrees(neuron_dim, neuron_dim, currentrotated_images + offset2, currentrotated_images + offset3);
    }

    // Rotate images
    #pragma omp parallel for
    for (int i = 1; i < num_real_rot; ++i) {
        for (int c = 0; c < spacing; ++c) {
            T *currentImage = image + c*image_size;
            T *currentRotatedImage = rotated_images + (i*spacing + c)*neuron_size;
            rotateAndCrop(image_dim, image_dim, neuron_dim, neuron_dim, currentImage, currentRotatedImage, i*angleStepRadians, interpolation);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage, currentRotatedImage + offset1);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage + offset1, currentRotatedImage + offset2);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage + offset2, currentRotatedImage + offset3);
        }
    }

    // Flip images
    if (use_flip)
    {
        T *flippedrotated_images = rotated_images + spacing * num_rot * neuron_size;

        #pragma omp parallel for
        for (int i = 0; i < num_rot; ++i) {
            for (int c = 0; c < spacing; ++c) {
                flip(neuron_dim, neuron_dim, rotated_images + (i*spacing + c)*neuron_size,
                    flippedrotated_images + (i*spacing + c)*neuron_size);
            }
        }
    }

    return rotated_images;
}

} // namespace pink
