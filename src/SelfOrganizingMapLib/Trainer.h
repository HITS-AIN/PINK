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
#include "DataIO.h"
#include "ImageProcessingLib/Interpolation.h"
#include "SelfOrganizingMap.h"
#include "SOM.h"
#include "UtilitiesLib/DistanceFunctor.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T>
class TrainerBase
{
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
       update_info(som_layout)
    {
        if (number_of_rotations <= 0 or (number_of_rotations != 1 and number_of_rotations % 4 != 0))
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

    float max_update_distance;
    Interpolation interpolation;

    /// Counting updates of each neuron
    UpdateInfoType update_info;
};

/// Primary template will never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
class Trainer;

/// GPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, true>  : public TrainerBase<SOMLayout, DataLayout, T>
{};

/// CPU version of training
template <typename SOMLayout, typename DataLayout, typename T>
class Trainer<SOMLayout, DataLayout, T, false>  : public TrainerBase<SOMLayout, DataLayout, T>
{
    typedef SOM<SOMLayout, DataLayout, T> SOMType;
    typedef typename TrainerBase<SOMLayout, DataLayout, T>::UpdateInfoType UpdateInfoType;

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
        std::vector<T> rotatedImages(rotatedImagesSize);
        std::vector<T> euclideanDistanceMatrix(som_size);
        std::vector<uint32_t> bestRotationMatrix(som_size);

        generateRotatedImages(&rotatedImages[0], const_cast<float*>(data.get_data_pointer()), this->number_of_rotations,
            data.get_dimension()[0], som.get_neuron_dimension()[0], this->use_flip, this->interpolation, 1);

        generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
            som_size, som.get_data_pointer(), neuron_size, numberOfRotationsAndFlip, &rotatedImages[0]);

        uint32_t best_match = findBestMatchingNeuron(&euclideanDistanceMatrix[0], som_size);

        float *current_neuron = som.get_data_pointer();
        for (int i = 0; i < som_size; ++i) {
            float distance = CartesianDistanceFunctor<2, false>(som.get_som_dimension()[0], som.get_som_dimension()[1])(best_match, i);
            if (this->max_update_distance <= 0 or distance < this->max_update_distance) {
                float factor = this->distribution_function(distance);
                float *current_image = &rotatedImages[0] + bestRotationMatrix[i] * neuron_size;
                for (int j = 0; j < neuron_size; ++j) {
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

} // namespace pink
