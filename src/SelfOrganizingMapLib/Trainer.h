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
#include "generate_euclidean_distance_matrix.h"
#include "generate_rotated_images.h"
#include "SOM.h"
#include "SOMIO.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/Interpolation.h"
#include "UtilitiesLib/pink_exception.h"

#ifdef __CUDACC__
    #include <thrust/host_vector.h>
    #include "CudaLib/CudaLib.h"
    #include "CudaLib/generate_euclidean_distance_matrix.h"
    #include "CudaLib/generate_rotated_images.h"
    #include "CudaLib/update_neurons.h"
#endif

//#define PRINT_DEBUG

namespace pink {

/// Abstract base class
struct TrainerBase
{
    virtual ~TrainerBase() {}

    virtual void update_som() = 0;
};

template <typename SOMLayout, typename DataLayout, typename T>
class TrainerCommon
{
public:

    TrainerCommon(SOM<SOMLayout, DataLayout, T> const& som, std::function<float(float)> const& distribution_function,
        int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim, EuclideanDistanceShape const& euclidean_distance_shape)
     : m_distribution_function(distribution_function),
       m_verbosity(verbosity),
       m_number_of_rotations(number_of_rotations),
       m_use_flip(use_flip),
       m_number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
       m_max_update_distance(max_update_distance),
       m_interpolation(interpolation),
       m_update_info(som.get_som_layout()),
       m_som_size(static_cast<uint32_t>(som.get_som_layout().size())),
       m_update_factors(m_som_size * m_som_size, 0.0),
       m_euclidean_distance_dim(euclidean_distance_dim),
       m_euclidean_distance_shape(euclidean_distance_shape)
    {
        if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
            throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");

        for (uint32_t i = 0; i < m_som_size; ++i) {
            for (uint32_t j = 0; j < m_som_size; ++j) {
                float distance = som.get_som_layout().get_distance(i, j);
                if (this->m_max_update_distance <= 0 or distance < this->m_max_update_distance) {
                    m_update_factors[i * m_som_size + j] = distribution_function(distance);
                }
            }
        }
    }

    auto get_update_info() const { return m_update_info; }

protected:

    typedef Data<SOMLayout, uint32_t> UpdateInfoType;

    std::function<float(float)> m_distribution_function;
    int m_verbosity;
    uint32_t m_number_of_rotations;
    bool m_use_flip;
    uint32_t m_number_of_spatial_transformations;

    float m_max_update_distance;
    Interpolation m_interpolation;

    /// Counting updates of each neuron
    UpdateInfoType m_update_info;

    /// Number of neurons
    uint32_t m_som_size;

    /// Pre-calculation of updating factors
    std::vector<float> m_update_factors;

    /// Dimension for calculation of euclidean distance
    uint32_t m_euclidean_distance_dim;

    /// Shape of euclidean distance region
    EuclideanDistanceShape m_euclidean_distance_shape;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Trainer;


/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, false> : public TrainerBase, public TrainerCommon<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerCommon<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer(SOMType& som, std::function<float(float)> const& distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        EuclideanDistanceShape const& euclidean_distance_shape = EuclideanDistanceShape::QUADRATIC)
     : TrainerCommon<SOMLayout, DataLayout, T>(som, distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, euclidean_distance_dim, euclidean_distance_shape),
       m_som(som)
    {}

    void operator () (Data<DataLayout, T> const& data)
    {
        auto&& spatial_transformed_images = SpatialTransformer<DataLayout>()(data, this->m_number_of_rotations,
            this->m_use_flip, this->m_interpolation, this->m_som.get_neuron_layout());

#ifdef PRINT_DEBUG
        std::cout << "spatial_transformed_images" << std::endl;
        for (auto&& e : spatial_transformed_images) std::cout << e << " ";
        std::cout << std::endl;
#endif

        // Memory allocation
        std::vector<T> euclidean_distance_matrix(this->m_som.get_number_of_neurons());
        std::vector<uint32_t> best_rotation_matrix(this->m_som.get_number_of_neurons());

        generate_euclidean_distance_matrix(euclidean_distance_matrix, best_rotation_matrix,
            this->m_som.get_number_of_neurons(), m_som.get_data_pointer(),
            m_som.get_neuron_layout(), this->m_number_of_spatial_transformations,
            spatial_transformed_images, this->m_euclidean_distance_dim, this->m_euclidean_distance_shape);

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

        auto neuron_size = m_som.get_neuron_size();
        auto&& current_neuron = m_som.get_data_pointer();
        for (uint32_t i = 0; i < this->m_som.get_number_of_neurons(); ++i) {
            float factor = this->m_update_factors[
                static_cast<size_t>(best_match * this->m_som.get_number_of_neurons()) + i];
            if (factor != 0.0f) {
                T *current_image = &spatial_transformed_images[best_rotation_matrix[i] * neuron_size];
                for (uint32_t j = 0; j < neuron_size; ++j) {
                    current_neuron[j] -= (current_neuron[j] - current_image[j]) * factor;
                }
            }
            current_neuron += neuron_size;
        }

        ++this->m_update_info[static_cast<uint32_t>(best_match)];

#ifdef PRINT_DEBUG
        std::cout << "best_match = " << best_match << std::endl;
#endif
    }

    void update_som()
    {}

private:

    /// A reference to the SOM will be trained
    SOMType& m_som;
};


#ifdef __CUDACC__

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true> : public TrainerBase, public TrainerCommon<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerCommon<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

public:

    Trainer(Trainer const&) = delete;

    Trainer(SOMType& som, std::function<float(float)> const& distribution_function, int verbosity,
        uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        EuclideanDistanceShape const& euclidean_distance_shape = EuclideanDistanceShape::QUADRATIC,
        uint32_t block_size = 256, DataType euclidean_distance_type = DataType::FLOAT)
     : TrainerCommon<SOMLayout, DataLayout, T>(som, distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, euclidean_distance_dim, euclidean_distance_shape),
       m_som(som),
       d_som(som.get_data()),
       m_block_size(block_size),
       m_euclidean_distance_type(euclidean_distance_type),
       d_spatial_transformed_images(this->m_number_of_spatial_transformations * som.get_neuron_size()),
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

        d_update_factors = this->m_update_factors;

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
    void operator () (Data<DataLayout, T> const& data)
    {
        /// Device memory for data
        thrust::device_vector<T> d_data = data.get_data();

        SpatialTransformerGPU<DataLayout>()(
            d_spatial_transformed_images, d_data,
            this->m_number_of_rotations, this->m_use_flip, this->m_interpolation,
            data.get_layout(),
            this->m_som.get_neuron_layout(),
            d_cos_alpha, d_sin_alpha);

#ifdef PRINT_DEBUG
        std::cout << "spatial_transformed_images" << std::endl;
        thrust::host_vector<T> spatial_transformed_images = d_spatial_transformed_images;
        for (auto&& e : spatial_transformed_images) std::cout << e << " ";
        std::cout << std::endl;
#endif

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            this->m_som.get_number_of_neurons(), this->m_som.get_neuron_layout(), d_som, this->m_number_of_spatial_transformations,
            d_spatial_transformed_images, m_block_size, m_euclidean_distance_type, this->m_euclidean_distance_dim,
            this->m_euclidean_distance_shape, d_circle_offset, d_circle_delta);

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
            d_best_match, d_update_factors, this->m_som.get_number_of_neurons(), this->m_som.get_neuron_layout().size());

        thrust::host_vector<uint32_t> best_match = d_best_match;
        ++this->m_update_info[best_match[0]];
    }

    void update_som()
    {
        thrust::copy(d_som.begin(), d_som.end(), m_som.get_data_pointer());
    }

private:

    /// A reference to the SOM will be trained
    SOMType& m_som;

    /// Device memory for SOM
    thrust::device_vector<T> d_som;

    uint32_t m_block_size;

    /// The data type for the euclidean distance
    DataType m_euclidean_distance_type;

    thrust::device_vector<T> d_spatial_transformed_images;
    thrust::device_vector<T> d_euclidean_distance_matrix;
    thrust::device_vector<uint32_t> d_best_rotation_matrix;
    thrust::device_vector<uint32_t> d_best_match;

    thrust::device_vector<float> d_cos_alpha;
    thrust::device_vector<float> d_sin_alpha;
    thrust::device_vector<float> d_update_factors;

    thrust::device_vector<uint32_t> d_circle_offset;
    thrust::device_vector<uint32_t> d_circle_delta;
};

#endif

} // namespace pink
