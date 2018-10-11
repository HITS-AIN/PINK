/**
 * @file   SelfOrganizingMapLib/TrainerGPU.h
 * @date   Oct 11, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include "ImageProcessingLib/CropAndRotate.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

class TrainerGPU
{
public:

    TrainerGPU(std::function<float(float)> distribution_function, int verbosity = 0,
        int number_of_rotations = 360, bool use_flip = true,
        float progress_factor = 0.1, int max_update_distance = 0)
     : distribution_function(distribution_function),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       progress_factor(progress_factor),
       max_update_distance(max_update_distance)
    {
        if (number_of_rotations <= 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");
    }

    template <typename SOMType>
    void operator () (SOMType& som, typename SOMType::NeuronType const& image) const
    {
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

        generateRotatedImages(&rotatedImages[0], const_cast<float*>(image.get_data_pointer()), number_of_rotations,
            image.get_dimension()[0], som.get_neuron_dimension()[0], use_flip, Interpolation::BILINEAR, 1);

        generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
            som_size, som.get_data_pointer(), neuron_size, numberOfRotationsAndFlip, &rotatedImages[0]);

        int bestMatch = findBestMatchingNeuron(&euclideanDistanceMatrix[0], som_size);

        float *current_neuron = som.get_data_pointer();
        for (int i = 0; i < som_size; ++i) {
            float distance = CartesianDistanceFunctor<2, false>(som.get_som_dimension()[0], som.get_som_dimension()[1])(bestMatch, i);
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
    int verbosity;
    int number_of_rotations;
    bool use_flip;
    float progress_factor;
    int max_update_distance;

};

} // namespace pink
