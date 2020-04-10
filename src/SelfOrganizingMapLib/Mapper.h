/**
 * @file   SelfOrganizingMapLib/Mapper.h
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

/// Abstract base class
struct MapperBase
{
    virtual ~MapperBase() {}
};

template <typename SOMLayout, typename DataLayout, typename T>
class MapperCommon
{
public:

    MapperCommon(SOM<SOMLayout, DataLayout, T> const& som, int verbosity, uint32_t number_of_rotations,
        bool use_flip, Interpolation interpolation, uint32_t euclidean_distance_dim,
        EuclideanDistanceShape const& euclidean_distance_shape)
     : m_som(som),
       m_verbosity(verbosity),
       m_number_of_rotations(number_of_rotations),
       m_use_flip(use_flip),
       m_number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
       m_angle_step_radians(static_cast<float>(2.0 * M_PI) / number_of_rotations),
       m_interpolation(interpolation),
       m_euclidean_distance_dim(euclidean_distance_dim),
       m_euclidean_distance_shape(euclidean_distance_shape)
    {
        if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
        {
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");
        }
        if (euclidean_distance_dim > m_som.get_neuron_layout().get_last_dimension())
        {
            throw pink::exception("Euclidean distance dimension larger than neuron dimension");
        }
    }

protected:

    /// A reference to the SOM will be trained
    SOM<SOMLayout, DataLayout, T> const& m_som;

    int m_verbosity;
    uint32_t m_number_of_rotations;
    bool m_use_flip;
    uint32_t m_number_of_spatial_transformations;
    float m_angle_step_radians;

    Interpolation m_interpolation;

    /// Dimension for calculation of euclidean distance
    uint32_t m_euclidean_distance_dim;

    /// Shape of euclidean distance region
    EuclideanDistanceShape m_euclidean_distance_shape;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Mapper;

/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Mapper<SOMLayout, DataLayout, T, false> : public MapperBase, public MapperCommon<SOMLayout, DataLayout, T>
{
public:

    Mapper(SOM<SOMLayout, DataLayout, T> const& som, int verbosity,
        uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        EuclideanDistanceShape const& euclidean_distance_shape = EuclideanDistanceShape::QUADRATIC)
     : MapperCommon<SOMLayout, DataLayout, T>(som, verbosity, number_of_rotations,
        use_flip, interpolation, euclidean_distance_dim, euclidean_distance_shape)
    {}

    auto operator () (Data<DataLayout, T> const& data)
    {
        auto&& spatial_transformed_images = SpatialTransformer<DataLayout>()(data, this->m_number_of_rotations,
            this->m_use_flip, this->m_interpolation, this->m_som.get_neuron_layout());

        std::vector<T> euclidean_distance_matrix(this->m_som.get_number_of_neurons());
        std::vector<uint32_t> best_rotation_matrix(this->m_som.get_number_of_neurons());

        generate_euclidean_distance_matrix(euclidean_distance_matrix, best_rotation_matrix,
            this->m_som.get_number_of_neurons(), this->m_som.get_data_pointer(),
            this->m_som.get_neuron_layout(), this->m_number_of_spatial_transformations,
            spatial_transformed_images, this->m_euclidean_distance_dim, this->m_euclidean_distance_shape);

        for (auto& e : euclidean_distance_matrix) e = std::sqrt(e);
        return std::make_tuple(euclidean_distance_matrix, best_rotation_matrix);
    }
};


#ifdef __CUDACC__

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Mapper<SOMLayout, DataLayout, T, true> : public MapperBase, public MapperCommon<SOMLayout, DataLayout, T>
{
public:

    Mapper(SOM<SOMLayout, DataLayout, T> const& som, int verbosity, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        EuclideanDistanceShape const& euclidean_distance_shape = EuclideanDistanceShape::QUADRATIC,
        uint32_t block_size = 256, DataType euclidean_distance_type = DataType::FLOAT)
     : MapperCommon<SOMLayout, DataLayout, T>(som, verbosity, number_of_rotations,
        use_flip, interpolation, euclidean_distance_dim, euclidean_distance_shape),
       d_som(som.get_data()),
       m_block_size(block_size),
       m_euclidean_distance_type(euclidean_distance_type),
       d_spatial_transformed_images(this->m_number_of_spatial_transformations * som.get_neuron_size()),
       d_euclidean_distance_matrix(som.get_number_of_neurons()),
       d_best_rotation_matrix(som.get_number_of_neurons()),
       d_best_match(1)
    {
        if (number_of_rotations >= 4) {
            uint32_t num_real_rot = number_of_rotations / 4;
            std::vector<float> cos_alpha(num_real_rot - 1);
            std::vector<float> sin_alpha(num_real_rot - 1);

            for (uint32_t i = 0; i < num_real_rot - 1; ++i) {
                float angle = (i+1) * this->m_angle_step_radians;
                cos_alpha[i] = std::cos(angle);
                sin_alpha[i] = std::sin(angle);
            }

            d_cos_alpha = cos_alpha;
            d_sin_alpha = sin_alpha;
        }

        if (euclidean_distance_shape == EuclideanDistanceShape::CIRCULAR) {
            std::vector<uint32_t> delta(euclidean_distance_dim);
            std::vector<uint32_t> offset(euclidean_distance_dim + 1);

            delta[0] = std::sqrt(euclidean_distance_dim * 0.5 - std::pow(0.5, 2));
            offset[0] = 0;
            for (uint32_t i = 1; i < euclidean_distance_dim; ++i) {
                delta[i] = std::round(std::sqrt(euclidean_distance_dim * (i + 0.5) - std::pow((i + 0.5), 2)));
                offset[i] = offset[i - 1] + 2 * delta[i - 1];
            }
            offset[euclidean_distance_dim] = offset[euclidean_distance_dim - 1] + 2 * delta[euclidean_distance_dim - 1];

            d_circle_offset = offset;
            d_circle_delta = delta;
        }
    }

    /// Training the SOM by a single data point
    auto operator () (Data<DataLayout, T> const& data)
    {
        /// Device memory for data
        thrust::device_vector<T> d_data = data.get_data();

        SpatialTransformerGPU<DataLayout>()(
            d_spatial_transformed_images, d_data,
            this->m_number_of_rotations, this->m_use_flip, this->m_interpolation,
            data.get_layout(),
            this->m_som.get_neuron_layout(),
            d_cos_alpha, d_sin_alpha);

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            this->m_som.get_number_of_neurons(), this->m_som.get_neuron_layout(), d_som, this->m_number_of_spatial_transformations,
            d_spatial_transformed_images, m_block_size, m_euclidean_distance_type, this->m_euclidean_distance_dim,
            this->m_euclidean_distance_shape, d_circle_offset, d_circle_delta);

        std::vector<float> euclidean_distance_matrix(this->m_som.get_number_of_neurons());
        std::vector<uint32_t> best_rotation_matrix(this->m_som.get_number_of_neurons());

        thrust::copy(d_euclidean_distance_matrix.begin(),
            d_euclidean_distance_matrix.end(), &euclidean_distance_matrix[0]);
        thrust::copy(d_best_rotation_matrix.begin(),
            d_best_rotation_matrix.end(), &best_rotation_matrix[0]);

        for (auto& e : euclidean_distance_matrix) e = std::sqrt(e);
        return std::make_tuple(euclidean_distance_matrix, best_rotation_matrix);
    }

private:

    /// Device memory for SOM
    thrust::device_vector<T> d_som;

    uint32_t m_block_size;

    /// The data type for the euclidean distance
    DataType m_euclidean_distance_type;

    thrust::device_vector<T> d_spatial_transformed_images;
    thrust::device_vector<float> d_euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_rotation_matrix;
    thrust::device_vector<uint32_t> d_best_match;

    thrust::device_vector<float> d_cos_alpha;
    thrust::device_vector<float> d_sin_alpha;

    thrust::device_vector<uint32_t> d_circle_offset;
    thrust::device_vector<uint32_t> d_circle_delta;
};

#endif

} // namespace pink
