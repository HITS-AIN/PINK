/**
 * @file   SelfOrganizingMapLib/generate_rotated_images.h
 * @date   Oct 26, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdint>
#include <vector>

#include "Data.h"
#include "ImageProcessingLib/ImageProcessing.h"

namespace pink {

template <typename LayoutType, typename T>
auto generate_rotated_images(Data<LayoutType, T> const& data,
    uint32_t num_rot, bool use_flip, Interpolation interpolation)
{
    std::vector<T> rotated_images;

    int image_size = image_dim * image_dim;
    int neuron_size = neuron_dim * neuron_dim;

    int num_real_rot = num_rot/4;
    T angleStepRadians = 2.0 * M_PI / num_rot;

    int offset1 = num_real_rot * numberOfChannels * neuron_size;
    int offset2 = 2 * offset1;
    int offset3 = 3 * offset1;

    // Copy original image to first position of image array
    #pragma omp parallel for
    for (int c = 0; c < numberOfChannels; ++c) {
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
        for (int c = 0; c < numberOfChannels; ++c) {
            T *currentImage = image + c*image_size;
            T *currentRotatedImage = rotated_images + (i*numberOfChannels + c)*neuron_size;
            rotateAndCrop(image_dim, image_dim, neuron_dim, neuron_dim, currentImage, currentRotatedImage, i*angleStepRadians, interpolation);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage, currentRotatedImage + offset1);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage + offset1, currentRotatedImage + offset2);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage + offset2, currentRotatedImage + offset3);
        }
    }

    // Flip images
    if (useFlip)
    {
        T *flippedrotated_images = rotated_images + numberOfChannels * num_rot * neuron_size;

        #pragma omp parallel for
        for (int i = 0; i < num_rot; ++i) {
            for (int c = 0; c < numberOfChannels; ++c) {
                flip(neuron_dim, neuron_dim, rotated_images + (i*numberOfChannels + c)*neuron_size,
                    flippedrotated_images + (i*numberOfChannels + c)*neuron_size);
            }
        }
    }
    return rotated_images;
}

} // namespace pink
