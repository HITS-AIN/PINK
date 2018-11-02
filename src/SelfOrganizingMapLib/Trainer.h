/**
 * @file   SelfOrganizingMapLib/Trainer.h
 * @date   Oct 11, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "Data.h"
#include "find_best_match.h"
#include "generate_rotated_images.h"
#include "generate_euclidean_distance_matrix.h"
#include "ImageProcessingLib/Interpolation.h"
#include "SOM.h"
#include "UtilitiesLib/DistanceFunctor.h"
#include "UtilitiesLib/pink_exception.h"

#ifdef __CUDACC__
    #include "CudaLib/CudaLib.h"
    #include "CudaLib/generate_euclidean_distance_matrix.h"
    #include "CudaLib/generate_rotated_images.h"
    //#include "CudaLib/update_neurons.h"
#endif

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class TrainerBase
{
public:

    TrainerBase(std::function<float(float)> distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, uint32_t spatial_transformed_image_dim,
        float max_update_distance, Interpolation interpolation, SOMLayout const& som_layout)
     : distribution_function(distribution_function),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
       spatial_transformed_image_dim(spatial_transformed_image_dim),
       max_update_distance(max_update_distance),
       interpolation(interpolation),
       update_info(som_layout)
    {
        if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");
    }

    auto get_update_info() const { return update_info; }

protected:

    typedef Data<SOMLayout, uint32_t> UpdateInfoType;

    std::function<float(float)> distribution_function;
    int verbosity;
    uint32_t number_of_rotations;
    bool use_flip;
    uint32_t number_of_spatial_transformations;
    uint32_t spatial_transformed_image_dim;

    float max_update_distance;
    Interpolation interpolation;

    /// Counting updates of each neuron
    UpdateInfoType update_info;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Trainer;

#ifndef __CUDACC__

/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, false> : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer(SOMType& som, std::function<float(float)> distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, uint32_t spatial_transformed_image_dim, float max_update_distance,
        Interpolation interpolation)
     : TrainerBase<SOMLayout, DataLayout, T>(distribution_function, verbosity, number_of_rotations,
           use_flip, spatial_transformed_image_dim, max_update_distance, interpolation, som.get_som_layout()),
       som(som)
    {}

    void operator () (Data<DataLayout, T> const& data)
    {
        uint32_t som_size = som.get_som_dimension()[0] * som.get_som_dimension()[1];
        uint32_t neuron_size = this->spatial_transformed_image_dim * this->spatial_transformed_image_dim;

        // Memory allocation
        std::vector<T> euclidean_distance_matrix(som_size);
        std::vector<uint32_t> best_rotation_matrix(som_size);

        auto&& list_of_spatial_transformed_images = generate_rotated_images(data, this->number_of_rotations,
            this->use_flip, this->interpolation, this->spatial_transformed_image_dim);

        generate_euclidean_distance_matrix(euclidean_distance_matrix, best_rotation_matrix,
            som_size, som.get_data(), neuron_size, this->number_of_spatial_transformations,
            list_of_spatial_transformed_images);

        /// Find the best matching neuron, with the lowest euclidean distance
        auto&& best_match = std::distance(euclidean_distance_matrix.begin(),
            std::min_element(std::begin(euclidean_distance_matrix), std::end(euclidean_distance_matrix)));

        auto&& current_neuron = som.get_data_pointer();
        for (uint32_t i = 0; i < som_size; ++i) {
            float distance = CartesianDistanceFunctor<2, false>(som.get_som_dimension()[0], som.get_som_dimension()[1])(best_match, i);
            if (this->max_update_distance <= 0 or distance < this->max_update_distance) {
                float factor = this->distribution_function(distance);
                T *current_image = &list_of_spatial_transformed_images[best_rotation_matrix[i] * neuron_size];
                for (uint32_t j = 0; j < neuron_size; ++j) {
                    current_neuron[j] -= (current_neuron[j] - current_image[j]) * factor;
                }
            }
            current_neuron += neuron_size;
        }

        ++this->update_info[best_match];
    }

private:

    /// A reference to the SOM will be trained
    SOMType& som;
};

#else // __CUDACC__

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true> : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer(SOMType& som, std::function<float(float)> distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, uint32_t spatial_transformed_image_dim, float max_update_distance,
        Interpolation interpolation, uint16_t block_size, bool use_multiple_gpus)
     : TrainerBase<SOMLayout, DataLayout, T>(distribution_function, verbosity, number_of_rotations,
           use_flip, spatial_transformed_image_dim, max_update_distance, interpolation, som.get_som_layout()),
       som(som),
       block_size(block_size),
       use_multiple_gpus(use_multiple_gpus),
       d_list_of_spatial_transformed_images(this->number_of_spatial_transformations * som.get_neuron_size()),
       d_euclidean_distance_matrix(som.get_number_of_neurons()),
       d_best_rotation_matrix(som.get_number_of_neurons()),
       d_best_match(1)
    {
        std::vector<T> cos_alpha(number_of_rotations - 1);
        std::vector<T> sin_alpha(number_of_rotations - 1);

        float angle_step_radians = 0.5 * M_PI / number_of_rotations;
        for (uint32_t i = 0; i < number_of_rotations - 1; ++i) {
            float angle = (i+1) * angle_step_radians;
            cos_alpha[i] = std::cos(angle);
            sin_alpha[i] = std::sin(angle);
        }

        d_cosAlpha = cos_alpha;
        d_sinAlpha = sin_alpha;
    }

    /// Training the SOM by a single data point
    void operator () (Data<DataLayout, T> const& data)
    {
        auto&& image_dim = data.get_dimension()[0];
        auto&& neuron_dim = som.get_neuron_dimension()[0];

        generate_rotated_images(d_list_of_spatial_transformed_images, data, this->number_of_rotations,
            image_dim, neuron_dim, this->use_flip, this->interpolation, d_cosAlpha, d_sinAlpha);

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            som.get_number_of_neurons(), som.get_neuron_size(), som.get_device_vector(), this->number_of_spatial_transformations,
            d_list_of_spatial_transformed_images, block_size, use_multiple_gpus);

//		update_neurons(som.get_device_vector(), d_list_of_spatial_transformed_images,
//			d_best_rotation_matrix, d_euclidean_distance_matrix, d_best_match,
//			som_width, inputData.som_height, inputData.som_depth, inputData.som_size,
//			neuron_size, inputData.function, inputData.layout,
//			inputData.sigma, inputData.damping, max_update_distance,
//			inputData.usePBC, inputData.dimensionality);

        uint32_t best_match;
        thrust::copy(d_best_match.begin(), d_best_match.end(), &best_match);
        ++this->update_info[best_match];
    }

private:

    /// A reference to the SOM will be trained
    SOMType& som;

    uint16_t block_size;

    bool use_multiple_gpus;

    thrust::device_vector<T> d_list_of_spatial_transformed_images;
    thrust::device_vector<T> d_euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_rotation_matrix;
    thrust::device_vector<uint32_t> d_best_match;

    thrust::device_vector<T> d_cosAlpha;
    thrust::device_vector<T> d_sinAlpha;
};

#endif

} // namespace pink
