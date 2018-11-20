/**
 * @file   SelfOrganizingMapLib/Trainer_generic.h
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
#include "SOMIO.h"
#include "UtilitiesLib/pink_exception.h"

#ifdef __CUDACC__
    #include "CudaLib/CudaLib.h"
    #include "CudaLib/generate_euclidean_distance_matrix.h"
    #include "CudaLib/generate_rotated_images.h"
    #include "CudaLib/update_neurons.h"
#endif

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class TrainerBase_generic
{
public:

    TrainerBase_generic(std::function<float(float)> distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, SOMLayout const& som_layout)
     : distribution_function(distribution_function),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
       max_update_distance(max_update_distance),
       interpolation(interpolation),
       update_info(som_layout),
       som_size(som_layout.size()),
       update_factors(som_size * som_size, 0.0)
    {
        if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");

        for (uint32_t i = 0; i < som_size; ++i) {
            for (uint32_t j = 0; j < som_size; ++j) {
                float distance = som_layout.get_distance(i, j);
                if (this->max_update_distance <= 0 or distance < this->max_update_distance) {
                    update_factors[i * som_size + j] = distribution_function(distance);
                }
            }
        }
    }

    auto get_update_info() const { return update_info; }

protected:

    typedef Data<SOMLayout, uint32_t> UpdateInfoType;

    std::function<float(float)> distribution_function;
    int verbosity;
    uint32_t number_of_rotations;
    bool use_flip;
    uint32_t number_of_spatial_transformations;

    float max_update_distance;
    Interpolation interpolation;

    /// Counting updates of each neuron
    UpdateInfoType update_info;

    /// Number of neurons
    uint32_t som_size;

    /// Pre-calculation of updating factors
    std::vector<float> update_factors;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Trainer_generic;


/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer_generic<SOMLayout, DataLayout, T, false> : public TrainerBase_generic<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase_generic<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer_generic(SOMType& som, std::function<float(float)> distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation)
     : TrainerBase_generic<SOMLayout, DataLayout, T>(distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, som.get_som_layout()),
       som(som)
    {}

    void operator () (Data<DataLayout, T> const& data)
    {
        uint32_t som_size = som.get_som_dimension()[0] * som.get_som_dimension()[1];
        uint32_t neuron_dim = som.get_neuron_dimension()[0];
        uint32_t neuron_size = neuron_dim * neuron_dim;

        // Memory allocation
        std::vector<T> euclidean_distance_matrix(som_size);
        std::vector<uint32_t> best_rotation_matrix(som_size);

        auto&& spatial_transformed_images = generate_rotated_images(data, this->number_of_rotations,
            this->use_flip, this->interpolation, neuron_dim);

//        for (auto&& e : spatial_transformed_images) std::cout << e << " ";
//        std::cout << std::endl;

        generate_euclidean_distance_matrix(euclidean_distance_matrix, best_rotation_matrix,
            som_size, som.get_data_pointer(), neuron_size, this->number_of_spatial_transformations,
            spatial_transformed_images);

        /// Find the best matching neuron, with the lowest euclidean distance
        auto&& best_match = std::distance(euclidean_distance_matrix.begin(),
            std::min_element(std::begin(euclidean_distance_matrix), std::end(euclidean_distance_matrix)));

        auto&& current_neuron = som.get_data_pointer();
        for (uint32_t i = 0; i < som_size; ++i) {
            float factor = this->update_factors[best_match * som_size + i];
            if (factor != 0.0) {
                T *current_image = &spatial_transformed_images[best_rotation_matrix[i] * neuron_size];
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


#ifdef __CUDACC__

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer_generic<SOMLayout, DataLayout, T, true> : public TrainerBase_generic<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase_generic<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer_generic(SOMType& som, std::function<float(float)> distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint16_t block_size, bool use_multiple_gpus)
     : TrainerBase_generic<SOMLayout, DataLayout, T>(distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, som.get_som_layout()),
       som(som),
       d_som(som.get_data()),
       block_size(block_size),
       use_multiple_gpus(use_multiple_gpus),
       d_spatial_transformed_images(this->number_of_spatial_transformations * som.get_neuron_size()),
       d_euclidean_distance_matrix(som.get_number_of_neurons()),
       d_best_rotation_matrix(som.get_number_of_neurons()),
       d_best_match(1)
    {
        if (number_of_rotations >= 4) {
            std::vector<float> cos_alpha(number_of_rotations - 1);
            std::vector<float> sin_alpha(number_of_rotations - 1);

            uint32_t num_real_rot = number_of_rotations / 4;
            float angle_step_radians = 0.5 * M_PI / num_real_rot;
            for (uint32_t i = 0; i < num_real_rot - 1; ++i) {
                float angle = (i+1) * angle_step_radians;
                cos_alpha[i] = std::cos(angle);
                sin_alpha[i] = std::sin(angle);
            }

            d_cos_alpha = cos_alpha;
            d_sin_alpha = sin_alpha;
        }

        d_update_factors = this->update_factors;
    }

    /// Training the SOM by a single data point
    void operator () (Data<DataLayout, T> const& data)
    {
        /// Device memory for data
        thrust::device_vector<T> d_data = data.get_data();

        uint32_t som_size = som.get_som_dimension()[0] * som.get_som_dimension()[1];
        uint32_t neuron_dim = som.get_neuron_dimension()[0];
        uint32_t neuron_size = neuron_dim * neuron_dim;
        uint32_t spacing = data.get_layout().dimensionality > 2 ? data.get_dimension()[2] : 1;
        for (uint32_t i = 3; i < data.get_layout().dimensionality; ++i) spacing *= data.get_dimension()[i];

        generate_rotated_images(d_spatial_transformed_images, d_data, spacing, this->number_of_rotations,
            data.get_dimension()[0], neuron_dim, this->use_flip, this->interpolation, d_cos_alpha, d_sin_alpha);

//        thrust::host_vector<T> spatial_transformed_images = d_spatial_transformed_images;
//        for (auto&& e : spatial_transformed_images) std::cout << e << " ";
//        std::cout << std::endl;

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            som_size, neuron_size, d_som, this->number_of_spatial_transformations,
            d_spatial_transformed_images, block_size, use_multiple_gpus);

        update_neurons(d_som, d_spatial_transformed_images, d_best_rotation_matrix, d_euclidean_distance_matrix,
            d_best_match, d_update_factors, som_size, neuron_size);

        thrust::host_vector<uint32_t> best_match = d_best_match;
        ++this->update_info[best_match[0]];
    }

    void update_som()
    {
        thrust::copy(d_som.begin(), d_som.end(), som.get_data_pointer());
    }

private:

    /// A reference to the SOM will be trained
    SOMType& som;

    /// Device memory for SOM
    thrust::device_vector<T> d_som;

    uint16_t block_size;

    bool use_multiple_gpus;

    thrust::device_vector<T> d_spatial_transformed_images;
    thrust::device_vector<T> d_euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_rotation_matrix;
    thrust::device_vector<uint32_t> d_best_match;

    thrust::device_vector<float> d_cos_alpha;
    thrust::device_vector<float> d_sin_alpha;
    thrust::device_vector<float> d_update_factors;
};

#endif

} // namespace pink
