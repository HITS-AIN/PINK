/**
 * @file   SelfOrganizingMapLib/Trainer_gpu.h
 * @date   Oct 11, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include "CudaLib/CudaLib.h"
#include "CudaLib/generate_euclidean_distance_matrix.h"
#include "CudaLib/generate_rotated_images.h"
#include "ImageProcessingLib/CropAndRotate.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "Data.h"
#include "SelfOrganizingMap.h"
#include "SOM_gpu.h"
#include "Trainer.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true>
{
    typedef SOM<SOMLayout, DataLayout, T, true> SOMType;
    typedef Data<SOMLayout, uint32_t> UpdateCounterType;

public:

    Trainer(SOMType& som, std::function<float(float)> distribution_function, int verbosity = 0,
        int number_of_rotations = 360, bool use_flip = true, float max_update_distance = 0.0,
        Interpolation interpolation = Interpolation::BILINEAR, uint32_t block_size = 1,
        bool use_multiple_gpus = true)
     : som(som),
       som_size(som.get_som_layout().get_size()),
       neuron_size(som.get_neuron_layout().get_size()),
       block_size(block_size),
       use_multiple_gpus(use_multiple_gpus),
       distribution_function(distribution_function),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       number_of_rotations_and_flip(number_of_rotations * (use_flip ? 2 : 1)),
       max_update_distance(max_update_distance),
       interpolation(interpolation),
       d_list_of_spatial_transformed_images(number_of_rotations_and_flip * neuron_size),
       d_euclidean_distance_matrix(som_size),
       d_best_rotation_matrix(som_size),
       d_best_match(1),
       update_counter(som.get_som_layout(), 0)
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

    void operator () (Data<DataLayout, T> const& data)
    {
        thrust::device_vector<T> d_image(data.get_data());

        auto image_dim = data.get_dimension()[0];
        auto neuron_dim = som.get_neuron_dimension()[0];
        auto number_of_channels = som.get_neuron_layout().dimensionality == 2 ? 1 : som.get_neuron_dimension()[2];

        generate_rotated_images(d_list_of_spatial_transformed_images, d_image, number_of_rotations,
            image_dim, neuron_dim, use_flip, interpolation, d_cosAlpha, d_sinAlpha, number_of_channels);

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            som_size, som.get_device_vector(), neuron_size, number_of_rotations_and_flip,
            d_list_of_spatial_transformed_images, block_size, use_multiple_gpus);

//        update_neurons_gpu(som.get_device_vector(), d_list_of_spatial_transformed_images,
//            d_best_rotation_matrix, d_euclidean_distance_matrix, d_best_match,
//            inputData.som_width, inputData.som_height, inputData.som_depth, inputData.som_size,
//            neuron_size, inputData.function, inputData.layout,
//            inputData.sigma, inputData.damping, max_update_distance,
//            inputData.usePBC, inputData.dimensionality);
//
//        int best_match;
//        thrust::copy(d_best_match.begin(), d_best_match.end(), &best_match);
//        update_counter(best_match);
    }

private:

    /// A reference to the SOM will be trained
    SOMType& som;

    /// The number of neurons within the SOM
    uint32_t som_size;

    /// The number of data points within a neuron
    uint32_t neuron_size;

    uint32_t block_size;
    bool use_multiple_gpus;

    /// Counting updates of each neuron
    UpdateCounterType update_counter;

    std::function<float(float)> distribution_function;
    int verbosity;
    uint32_t number_of_rotations;
    bool use_flip;
    uint32_t number_of_rotations_and_flip;
    float max_update_distance;
    Interpolation interpolation;

    thrust::device_vector<T> d_list_of_spatial_transformed_images;
    thrust::device_vector<T> d_euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_rotation_matrix;
    thrust::device_vector<uint32_t> d_best_match;

    thrust::device_vector<T> d_cosAlpha;
    thrust::device_vector<T> d_sinAlpha;

};

} // namespace pink
