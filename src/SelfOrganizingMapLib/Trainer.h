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
#include "SOM.h"
#include "SOMIO.h"
#include "UtilitiesLib/Interpolation.h"
#include "UtilitiesLib/pink_exception.h"

#ifdef __CUDACC__
    #include "CudaLib/CudaLib.h"
    #include "CudaLib/generate_euclidean_distance_matrix.h"
    #include "CudaLib/generate_rotated_images.h"
    #include "CudaLib/update_neurons.h"
#endif

//#define PRINT_DEBUG

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class TrainerBase
{
public:

    TrainerBase(SOM<SOMLayout, DataLayout, T> const& som, std::function<float(float)> distribution_function,
        int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim)
     : distribution_function(distribution_function),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
       max_update_distance(max_update_distance),
       interpolation(interpolation),
       update_info(som.get_som_layout()),
       som_size(static_cast<uint32_t>(som.get_som_layout().size())),
       update_factors(som_size * som_size, 0.0),
       euclidean_distance_dim(euclidean_distance_dim)
    {
        if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");

        for (uint32_t i = 0; i < som_size; ++i) {
            for (uint32_t j = 0; j < som_size; ++j) {
                float distance = som.get_som_layout().get_distance(i, j);
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

    /// Dimension for calculation of euclidean distance
    uint32_t euclidean_distance_dim;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Trainer;


/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, false> : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer(SOMType& som, std::function<float(float)> const& distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim)
     : TrainerBase<SOMLayout, DataLayout, T>(som, distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, euclidean_distance_dim),
       som(som)
    {}

    void operator () (Data<DataLayout, T> const& data)
    {
        uint32_t neuron_dim = som.get_neuron_dimension()[0];
        uint32_t neuron_size = neuron_dim * neuron_dim;

        // Memory allocation
        std::vector<T> euclidean_distance_matrix(this->som.get_number_of_neurons());
        std::vector<uint32_t> best_rotation_matrix(this->som.get_number_of_neurons());

        auto&& spatial_transformed_images = generate_rotated_images(data, this->number_of_rotations,
            this->use_flip, this->interpolation, neuron_dim);

#ifdef PRINT_DEBUG
        std::cout << "spatial_transformed_images" << std::endl;
        for (auto&& e : spatial_transformed_images) std::cout << e << " ";
        std::cout << std::endl;
#endif

        generate_euclidean_distance_matrix(euclidean_distance_matrix, best_rotation_matrix,
            this->som.get_number_of_neurons(), som.get_data_pointer(),
            neuron_dim, this->number_of_spatial_transformations,
            spatial_transformed_images, this->euclidean_distance_dim);

#ifdef PRINT_DEBUG
        std::cout << "euclidean_distance_matrix" << std::endl;
        for (auto&& e : euclidean_distance_matrix) std::cout << e << " ";
        std::cout << std::endl;

        std::cout << "best_rotation_matrix" << std::endl;
        for (auto&& e : best_rotation_matrix) std::cout << e << " ";
        std::cout << std::endl;
#endif

        /// Find the best matching neuron, with the lowest euclidean distance
        auto&& best_match = std::distance(euclidean_distance_matrix.begin(),
            std::min_element(std::begin(euclidean_distance_matrix), std::end(euclidean_distance_matrix)));

        auto&& current_neuron = som.get_data_pointer();
        for (uint32_t i = 0; i < this->som.get_number_of_neurons(); ++i) {
            float factor = this->update_factors[
                static_cast<size_t>(best_match * this->som.get_number_of_neurons()) + i];
            if (factor != 0.0f) {
                T *current_image = &spatial_transformed_images[best_rotation_matrix[i] * neuron_size];
                for (uint32_t j = 0; j < neuron_size; ++j) {
                    current_neuron[j] -= (current_neuron[j] - current_image[j]) * factor;
                }
            }
            current_neuron += neuron_size;
        }

        ++this->update_info[static_cast<uint32_t>(best_match)];
    }

private:

    /// A reference to the SOM will be trained
    SOMType& som;
};


#ifdef __CUDACC__

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true> : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer(SOMType& som, std::function<float(float)> const& distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        uint16_t block_size = 256, DataType euclidean_distance_type = DataType::FLOAT)
     : TrainerBase<SOMLayout, DataLayout, T>(som, distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, euclidean_distance_dim),
       som(som),
       d_som(som.get_data()),
       block_size(block_size),
       euclidean_distance_type(euclidean_distance_type),
       d_spatial_transformed_images(this->number_of_spatial_transformations * som.get_neuron_size()),
       d_euclidean_distance_matrix(som.get_number_of_neurons()),
       d_best_rotation_matrix(som.get_number_of_neurons()),
       d_best_match(1)
    {
        if (number_of_rotations >= 4) {
            std::vector<float> cos_alpha(number_of_rotations - 1);
            std::vector<float> sin_alpha(number_of_rotations - 1);

            uint32_t num_real_rot = number_of_rotations / 4;
            float angle_step_radians = static_cast<float>(0.5 * M_PI) / num_real_rot;
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

        uint32_t neuron_dim = som.get_neuron_dimension()[0];
        uint32_t neuron_size = neuron_dim * neuron_dim;
        uint32_t spacing = data.get_layout().dimensionality > 2 ? data.get_dimension()[2] : 1;
        for (uint32_t i = 3; i < data.get_layout().dimensionality; ++i) spacing *= data.get_dimension()[i];

        generate_rotated_images(d_spatial_transformed_images, d_data, spacing, this->number_of_rotations,
            data.get_dimension()[0], neuron_dim, this->use_flip, this->interpolation, d_cos_alpha, d_sin_alpha);

#ifdef PRINT_DEBUG
        std::cout << "spatial_transformed_images" << std::endl;
        thrust::host_vector<T> spatial_transformed_images = d_spatial_transformed_images;
        for (auto&& e : spatial_transformed_images) std::cout << e << " ";
        std::cout << std::endl;
#endif

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            this->som.get_number_of_neurons(), neuron_size, d_som, this->number_of_spatial_transformations,
            d_spatial_transformed_images, block_size, euclidean_distance_type, this->euclidean_distance_dim);

#ifdef PRINT_DEBUG
        std::cout << "euclidean_distance_matrix" << std::endl;
        thrust::host_vector<T> euclidean_distance_matrix = d_euclidean_distance_matrix;
        for (auto&& e : euclidean_distance_matrix) std::cout << e << " ";
        std::cout << std::endl;

        std::cout << "best_rotation_matrix" << std::endl;
        thrust::host_vector<T> best_rotation_matrix = d_best_rotation_matrix;
        for (auto&& e : best_rotation_matrix) std::cout << e << " ";
        std::cout << std::endl;
#endif

        update_neurons(d_som, d_spatial_transformed_images, d_best_rotation_matrix, d_euclidean_distance_matrix,
            d_best_match, d_update_factors, this->som.get_number_of_neurons(), neuron_size);

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

    /// The data type for the euclidean distance
    DataType euclidean_distance_type;

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
