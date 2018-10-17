/**
 * @file   SelfOrganizingMapLib/TrainerGPU.h
 * @date   Oct 11, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include "CudaLib/CudaLib.h"
#include "ImageProcessingLib/CropAndRotate.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "Trainer.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true>
{
public:

    Trainer(std::function<float(float)> distribution_function, uint32_t image_dim, uint32_t neuron_dim,
        uint32_t number_of_channels, int verbosity = 0, int number_of_rotations = 360, bool use_flip = true,
        int max_update_distance = 0, Interpolation interpolation = Interpolation::BILINEAR)
     : distribution_function(distribution_function),
       image_dim(image_dim),
       neuron_dim(neuron_dim),
       number_of_channels(number_of_channels),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       max_update_distance(max_update_distance),
       interpolation(interpolation),
       d_list_of_spatial_transformed_images(number_of_rotations)
    {
        if (number_of_rotations <= 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");

        std::vector<T> cos_alpha(number_of_rotations - 1);
        std::vector<T> sin_alpha(number_of_rotations - 1);

        float angle_step_radians = 0.5 * M_PI / number_of_rotations;
        for (int i = 0; i < number_of_rotations - 1; ++i) {
            float angle = (i+1) * angle_step_radians;
            cos_alpha[i] = std::cos(angle);
            sin_alpha[i] = std::sin(angle);
        }

        d_cosAlpha = cos_alpha;
        d_sinAlpha = sin_alpha;
    }

    void operator () (SOM<SOMLayout, DataLayout, T>& som, Data<DataLayout, T> const& data) const
    {
        thrust::device_vector<T> d_image(data.get_data());

        generate_rotated_images_gpu(d_list_of_spatial_transformed_images, d_image, number_of_rotations,
            image_dim, neuron_dim, use_flip, interpolation, d_cosAlpha, d_sinAlpha, number_of_channels);

        int som_size = som.get_som_dimension()[0] * som.get_som_dimension()[1];
        int neuron_size = som.get_neuron_dimension()[0] * som.get_neuron_dimension()[1];
        int numberOfRotationsAndFlip = number_of_rotations;
        if (use_flip) numberOfRotationsAndFlip *= 2;
        int rotatedImagesSize = numberOfRotationsAndFlip * neuron_size;

        if (verbosity) std::cout << "som_size = " << som_size << "\n"
                                 << "neuron_size = " << neuron_size << "\n"
                                 << "number_of_rotations = " << number_of_rotations << "\n"
                                 << "numberOfRotationsAndFlip = " << numberOfRotationsAndFlip << "\n"
                                 << "rotatedImagesSize = " << rotatedImagesSize << std::endl;

        // Memory allocation
        std::vector<float> rotatedImages(rotatedImagesSize);
        std::vector<float> euclideanDistanceMatrix(som_size);
        std::vector<int> bestRotationMatrix(som_size);

        generateRotatedImages(&rotatedImages[0], const_cast<float*>(data.get_data_pointer()), number_of_rotations,
            data.get_dimension()[0], som.get_neuron_dimension()[0], use_flip, interpolation, 1);

        generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
            som_size, som.get_data_pointer(), neuron_size, numberOfRotationsAndFlip, &rotatedImages[0]);

        int bestMatch = findBestMatchingNeuron(&euclideanDistanceMatrix[0], som_size);

        float *current_neuron = som.get_data_pointer();
        for (int i = 0; i < som_size; ++i) {
            float distance = 1.0; // CartesianDistanceFunctor<2, false>(som.get_som_dimension()[0], som.get_som_dimension()[1])(bestMatch, i);
            if (max_update_distance <= 0 or distance < max_update_distance) {
                float factor = distribution_function(distance);
                float *current_image = &rotatedImages[0] + bestRotationMatrix[i] * neuron_size;
                for (int j = 0; j < neuron_size; ++j) {
                    current_neuron[j] -= (current_neuron[j] - current_image[j]) * factor;
                }
            }
            current_neuron += neuron_size;
        }

//		auto&& list_of_spatial_transformed_images = SpatialTransformer(Rotation<0,1>(number_of_rotations), use_flip)(image);
//		auto&& [euclidean_distance] generate_euclidean_distance_matrix(som, list_of_spatial_transformed_images);
//
//		auto&& best_match = find_best_match();
//
//		update_counter(best_match);
//		update_neurons(best_match);
    }

private:

    std::function<float(float)> distribution_function;
    uint32_t image_dim;
    uint32_t neuron_dim;
    uint32_t number_of_channels;
    int verbosity;
    int number_of_rotations;
    bool use_flip;
    int max_update_distance;
    Interpolation interpolation;

    thrust::device_vector<T> d_list_of_spatial_transformed_images;
    thrust::device_vector<T> d_cosAlpha;
    thrust::device_vector<T> d_sinAlpha;

};

} // namespace pink
