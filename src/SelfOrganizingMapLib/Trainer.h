/**
 * @file   SelfOrganizingMapLib/Trainer.h
 * @date   Oct 11, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "Data.h"
#include "ImageProcessingLib/Interpolation.h"
#include "SelfOrganizingMap.h"
#include "SOM.h"
#include "UtilitiesLib/pink_exception.h"

#ifdef __CUDACC__
    #include "CudaLib/CudaLib.h"
    #include "CudaLib/generate_euclidean_distance_matrix.h"
    #include "CudaLib/generate_rotated_images.h"
#endif

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class TrainerBase
{
    typedef Data<SOMLayout, uint32_t> UpdateCounterType;

public:

	TrainerBase(std::function<float(float)> distribution_function, int verbosity,
		int number_of_rotations, bool use_flip, float max_update_distance,
		Interpolation interpolation, SOMLayout const& som_layout)
	 : distribution_function(distribution_function),
	   verbosity(verbosity),
	   number_of_rotations(number_of_rotations),
	   use_flip(use_flip),
	   number_of_spatial_transformations(number_of_rotations * (use_flip ? 2 : 1)),
	   max_update_distance(max_update_distance),
	   interpolation(interpolation),
	   update_counter(som_layout)
	{
		if (number_of_rotations <= 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
			throw pink::exception("Number of rotations must be 1 or larger then 1 and divisible by 4");
	}

protected:

    std::function<float(float)> distribution_function;
    int verbosity;
    uint32_t number_of_rotations;
    bool use_flip;
    uint32_t number_of_spatial_transformations;

    float max_update_distance;
    Interpolation interpolation;

    /// Counting updates of each neuron
    UpdateCounterType update_counter;
};

/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU = false>
class Trainer : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T, false> SOMType;
    typedef Data<SOMLayout, uint32_t> UpdateCounterType;

public:

    Trainer(SOMType& som, std::function<float(float)> distribution_function, int verbosity = 0,
        int number_of_rotations = 360, bool use_flip = true, float max_update_distance = 0.0,
        Interpolation interpolation = Interpolation::BILINEAR)
     : TrainerBase<SOMLayout, DataLayout, T>(distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, som.get_som_layout()),
	   som(som)
    {}

    void operator () (Data<DataLayout, T> const& data)
    {
        int som_size = som.get_som_dimension()[0] * som.get_som_dimension()[1];
        int neuron_size = som.get_neuron_dimension()[0] * som.get_neuron_dimension()[1];
        int numberOfRotationsAndFlip = this->number_of_rotations;
        if (this->use_flip) numberOfRotationsAndFlip *= 2;
        int rotatedImagesSize = numberOfRotationsAndFlip * neuron_size;

        if (this->verbosity) std::cout << "som_size = " << som_size << "\n"
                                 << "neuron_size = " << neuron_size << "\n"
                                 << "number_of_rotations = " << this->number_of_rotations << "\n"
                                 << "numberOfRotationsAndFlip = " << numberOfRotationsAndFlip << "\n"
                                 << "rotatedImagesSize = " << rotatedImagesSize << std::endl;

        // Memory allocation
        std::vector<float> rotatedImages(rotatedImagesSize);
        std::vector<float> euclideanDistanceMatrix(som_size);
        std::vector<int> bestRotationMatrix(som_size);

        generateRotatedImages(&rotatedImages[0], const_cast<float*>(data.get_data_pointer()), this->number_of_rotations,
            data.get_dimension()[0], som.get_neuron_dimension()[0], this->use_flip, this->interpolation, 1);

        generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
            som_size, som.get_data_pointer(), neuron_size, numberOfRotationsAndFlip, &rotatedImages[0]);

        int bestMatch = findBestMatchingNeuron(&euclideanDistanceMatrix[0], som_size);

        float *current_neuron = som.get_data_pointer();
        for (int i = 0; i < som_size; ++i) {
            float distance = CartesianDistanceFunctor<2, false>(som.get_som_dimension()[0], som.get_som_dimension()[1])(bestMatch, i);
            if (this->max_update_distance <= 0 or distance < this->max_update_distance) {
                float factor = this->distribution_function(distance);
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

    /// A reference to the SOM will be trained
    SOMType& som;
};

#ifdef __CUDACC__
/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true> : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T, true> SOMType;
    typedef Data<SOMLayout, uint32_t> UpdateCounterType;

public:

    Trainer(SOMType& som, std::function<float(float)> distribution_function, int verbosity = 0,
        int number_of_rotations = 360, bool use_flip = true, float max_update_distance = 0.0,
        Interpolation interpolation = Interpolation::BILINEAR, uint16_t block_size = 1,
        bool use_multiple_gpus = true)
     : TrainerBase<SOMLayout, DataLayout, T>(distribution_function, verbosity, number_of_rotations,
           use_flip, max_update_distance, interpolation, som.get_som_layout()),
       som(som),
       block_size(block_size),
	   use_multiple_gpus(use_multiple_gpus),
       d_list_of_spatial_transformed_images(this->number_of_spatial_transformations * neuron_size),
       d_euclidean_distance_matrix(som.get_number_of_neurons()),
       d_best_rotation_matrix(som.get_number_of_neurons()),
       d_best_match(1)
    {
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

    /// Training the SOM by a single data point
    void operator () (Data<DataLayout, T> const& data)
    {
        thrust::device_vector<T> d_image(data.get_data());

        auto image_dim = data.get_dimension()[0];
        auto neuron_dim = som.get_neuron_dimension()[0];
        auto number_of_channels = som.get_neuron_layout().dimensionality == 2 ? 1 : som.get_neuron_dimension()[2];

        generate_rotated_images(d_list_of_spatial_transformed_images, d_image, this->number_of_rotations,
            image_dim, neuron_dim, this->use_flip, this->interpolation, d_cosAlpha, d_sinAlpha, number_of_channels);

        generate_euclidean_distance_matrix(d_euclidean_distance_matrix, d_best_rotation_matrix,
            som.get_number_of_neurons(), som.get_device_vector(), this->number_of_spatial_transformations,
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
