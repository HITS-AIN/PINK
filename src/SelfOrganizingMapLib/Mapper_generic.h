/**
 * @file   SelfOrganizingMapLib/Mapper_generic.h
 * @date   Nov 30, 2018
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

//#define PRINT_DEBUG

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class MapperBase_generic
{
public:

    MapperBase_generic(SOM<SOMLayout, DataLayout, T> const& som, int verbosity, uint32_t number_of_rotations,
        bool use_flip, Interpolation interpolation)
     : som(som),
       verbosity(verbosity),
       number_of_rotations(number_of_rotations),
       use_flip(use_flip),
       number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
       angle_step_radians(0.5 * M_PI / number_of_rotations / 4),
       interpolation(interpolation)
    {
        if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");
    }

protected:

    /// A reference to the SOM will be trained
    SOM<SOMLayout, DataLayout, T> const& som;

    int verbosity;
    uint32_t number_of_rotations;
    bool use_flip;
    uint32_t number_of_spatial_transformations;
    float angle_step_radians;

    Interpolation interpolation;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Mapper_generic;

/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Mapper_generic<SOMLayout, DataLayout, T, false> : public MapperBase_generic<SOMLayout, DataLayout, T>
{
public:

    Mapper_generic(SOM<SOMLayout, DataLayout, T> const& som, int verbosity,
        uint32_t number_of_rotations, bool use_flip, Interpolation interpolation)
     : MapperBase_generic<SOMLayout, DataLayout, T>(som, verbosity, number_of_rotations, use_flip, interpolation)
    {}

    auto operator () (Data<DataLayout, T> const& data)
    {
        uint32_t neuron_dim = this->som.get_neuron_dimension()[0];
        uint32_t neuron_size = neuron_dim * neuron_dim;

        auto&& spatial_transformed_images = generate_rotated_images(data, this->number_of_rotations,
            this->use_flip, this->interpolation, neuron_dim);

        std::vector<T> euclidean_distance_matrix(this->som.get_number_of_neurons());
        std::vector<uint32_t> best_rotation_matrix(this->som.get_number_of_neurons());

        generate_euclidean_distance_matrix(euclidean_distance_matrix, best_rotation_matrix,
            this->som.get_number_of_neurons(), this->som.get_data_pointer(), neuron_size, this->number_of_spatial_transformations,
            spatial_transformed_images);

        return std::make_tuple(euclidean_distance_matrix, best_rotation_matrix);
    }
};


#ifdef __CUDACC__

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Mapper_generic<SOMLayout, DataLayout, T, true> : public MapperBase_generic<SOMLayout, DataLayout, T>
{
public:

    Mapper_generic(SOM<SOMLayout, DataLayout, T> const& som, int verbosity, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, uint16_t block_size, bool use_multiple_gpus, DataType euclidean_distance_type)
     : MapperBase_generic<SOMLayout, DataLayout, T>(som, verbosity, number_of_rotations, use_flip, interpolation),
       d_som(som.get_data()),
       block_size(block_size),
       use_multiple_gpus(use_multiple_gpus),
       euclidean_distance_type(euclidean_distance_type),
       d_spatial_transformed_images(this->number_of_spatial_transformations * som.get_neuron_size()),
       d_euclidean_distance_matrix(som.get_number_of_neurons()),
       d_best_rotation_matrix(som.get_number_of_neurons()),
       d_best_match(1)
    {
        if (number_of_rotations >= 4) {
            uint32_t num_real_rot = number_of_rotations / 4;
            std::vector<float> cos_alpha(num_real_rot - 1);
            std::vector<float> sin_alpha(num_real_rot - 1);

            for (uint32_t i = 0; i < num_real_rot - 1; ++i) {
                float angle = (i+1) * this->angle_step_radians;
                cos_alpha[i] = std::cos(angle);
                sin_alpha[i] = std::sin(angle);
            }

            d_cos_alpha = cos_alpha;
            d_sin_alpha = sin_alpha;
        }
    }

    /// Training the SOM by a single data point
    auto operator () (Data<DataLayout, T> const& data)
    {
        /// Device memory for data
        thrust::device_vector<T> d_data = data.get_data();

        uint32_t neuron_dim = this->som.get_neuron_dimension()[0];
        uint32_t neuron_size = neuron_dim * neuron_dim;
        uint32_t spacing = data.get_layout().dimensionality > 2 ? data.get_dimension()[2] : 1;
        for (uint32_t i = 3; i < data.get_layout().dimensionality; ++i) spacing *= data.get_dimension()[i];

        generate_rotated_images(d_spatial_transformed_images, d_data, spacing, this->number_of_rotations,
            data.get_dimension()[0], neuron_dim, this->use_flip, this->interpolation, d_cos_alpha, d_sin_alpha);

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            this->som.get_number_of_neurons(), neuron_size, d_som, this->number_of_spatial_transformations,
            d_spatial_transformed_images, block_size, use_multiple_gpus, euclidean_distance_type);

        std::vector<T> euclidean_distance_matrix(this->som.get_number_of_neurons());
        std::vector<uint32_t> best_rotation_matrix(this->som.get_number_of_neurons());

        thrust::copy(d_euclidean_distance_matrix.begin(), d_euclidean_distance_matrix.end(), &euclidean_distance_matrix[0]);
        thrust::copy(d_best_rotation_matrix.begin(), d_best_rotation_matrix.end(), &best_rotation_matrix[0]);

        return std::make_tuple(euclidean_distance_matrix, best_rotation_matrix);
    }

private:

    /// Device memory for SOM
    thrust::device_vector<T> d_som;

    uint16_t block_size;

    bool use_multiple_gpus;

    /// The data type for the euclidean distance
    DataType euclidean_distance_type;

    thrust::device_vector<T> d_spatial_transformed_images;
    thrust::device_vector<T> d_euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_rotation_matrix;
    thrust::device_vector<uint32_t> d_best_match;

    thrust::device_vector<float> d_cos_alpha;
    thrust::device_vector<float> d_sin_alpha;
};

#endif

} // namespace pink
