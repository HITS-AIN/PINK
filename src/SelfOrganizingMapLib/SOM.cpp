/**
 * @file   SelfOrganizingMapLib/SOM.cpp
 * @brief  Self organizing Kohonen-map.
 * @date   Nov 25, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "SOM.h"
#include "UtilitiesLib/Error.h"
#include "UtilitiesLib/Filler.h"

namespace pink {

SOM::SOM(InputData const& inputData)
 : inputData_(inputData),
   som_(inputData.numberOfChannels * inputData.som_size * inputData.neuron_size),
   updateCounterMatrix_(inputData.som_size)
{
    // Initialize SOM
    if (inputData.init == SOMInitialization::ZERO)
        fillWithValue(&som_[0], som_.size());
    else if (inputData.init == SOMInitialization::RANDOM)
        fillWithRandomNumbers(&som_[0], som_.size(), inputData.seed);
    else if (inputData.init == SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION) {
        fillWithRandomNumbers(&som_[0], som_.size(), inputData.seed);
        for (int n = 0; n < inputData.som_size; ++n)
            for (int c = 0; c < inputData.numberOfChannels; ++c)
                for (int i = 0; i < inputData.neuron_dim; ++i)
                    som_[(n*inputData.numberOfChannels + c)*inputData.neuron_size + i*inputData.neuron_dim + i] = 1.0f;
    }
    else if (inputData.init == SOMInitialization::FILEINIT) {
        std::ifstream is(inputData.somFilename);
        if (!is) throw std::runtime_error("Error opening " + inputData.somFilename);

        int tmp;
        is.read((char*)&tmp, sizeof(int));
        if (tmp != inputData.numberOfChannels) throw std::runtime_error("readSOM: wrong numberOfChannels.");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != inputData.som_width) throw std::runtime_error("readSOM: wrong width.");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != inputData.som_height) throw std::runtime_error("readSOM: wrong height.");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != inputData.som_depth) throw std::runtime_error("readSOM: wrong depth.");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != inputData.neuron_dim) throw std::runtime_error("readSOM: wrong neuron_dim.");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != inputData.neuron_dim) throw std::runtime_error("readSOM: wrong neuron_dim.");
        is.read((char*)&som_[0], inputData.numberOfChannels * inputData.som_size * inputData.neuron_dim
            * inputData.neuron_dim * sizeof(float));
    } else
        fatalError("Unknown initType.");

    // Set distribution function
    if (inputData_.function == DistributionFunction::GAUSSIAN)
        ptrDistributionFunctor_ = std::make_shared<GaussianFunctor>(inputData_.sigma);
    else if (inputData_.function == DistributionFunction::MEXICANHAT)
        ptrDistributionFunctor_ = std::make_shared<MexicanHatFunctor>(inputData_.sigma);
    else
        fatalError("Unknown distribution function.");

    // Set distance function
    if (inputData_.layout == Layout::CARTESIAN) {
        if (inputData_.usePBC) {
            if (inputData_.dimensionality == 1) {
                ptrDistanceFunctor_ = std::make_shared<CartesianDistanceFunctor<1, true>>(inputData.som_width);
            } else if (inputData_.dimensionality == 2) {
                ptrDistanceFunctor_ = std::make_shared<CartesianDistanceFunctor<2, true>>(inputData.som_width, inputData.som_height);
            } else if (inputData_.dimensionality == 3) {
                ptrDistanceFunctor_ = std::make_shared<CartesianDistanceFunctor<3, true>>(inputData.som_width, inputData.som_height, inputData.som_depth);
            }
        } else {
            if (inputData_.dimensionality == 1) {
                ptrDistanceFunctor_ = std::make_shared<CartesianDistanceFunctor<1>>(inputData.som_width);
            } else if (inputData_.dimensionality == 2) {
                ptrDistanceFunctor_ = std::make_shared<CartesianDistanceFunctor<2>>(inputData.som_width, inputData.som_height);
            } else if (inputData_.dimensionality == 3) {
                ptrDistanceFunctor_ = std::make_shared<CartesianDistanceFunctor<3>>(inputData.som_width, inputData.som_height, inputData.som_depth);
            }
        }
    } else if (inputData_.layout == Layout::HEXAGONAL) {
        ptrDistanceFunctor_ = std::make_shared<HexagonalDistanceFunctor>(inputData.som_width);
    } else {
        fatalError("Unknown layout.");
    }
}

void SOM::write(std::string const& filename) const
{
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("Error opening " + filename);

    os.write((char*)&inputData_.numberOfChannels, sizeof(int));
    os.write((char*)&inputData_.som_width, sizeof(int));
    os.write((char*)&inputData_.som_height, sizeof(int));
    os.write((char*)&inputData_.som_depth, sizeof(int));
    os.write((char*)&inputData_.neuron_dim, sizeof(int));
    os.write((char*)&inputData_.neuron_dim, sizeof(int));
    os.write((char*)&som_[0], inputData_.numberOfChannels * inputData_.som_size
        * inputData_.neuron_dim * inputData_.neuron_dim * sizeof(float));
}

void SOM::updateNeurons(float *rotatedImages, int bestMatch, int *bestRotationMatrix)
{
    float distance, factor;
    float *current_neuron = &som_[0];

    for (int i = 0; i < inputData_.som_size; ++i) {
        distance = (*ptrDistanceFunctor_)(bestMatch, i);
        if (inputData_.maxUpdateDistance <= 0.0 or distance < inputData_.maxUpdateDistance) {
            factor = (*ptrDistributionFunctor_)(distance) * inputData_.damping;
            updateSingleNeuron(current_neuron, rotatedImages + bestRotationMatrix[i]
                * inputData_.numberOfChannels * inputData_.neuron_size, factor);
        }
        current_neuron += inputData_.numberOfChannels * inputData_.neuron_size;
    }
}

void SOM::printUpdateCounter() const
{
    if (inputData_.verbose) {
        std::cout << "\n  Number of updates of each neuron:\n" << std::endl;
        if (inputData_.layout == Layout::HEXAGONAL) {
            int radius = (inputData_.som_width - 1)/2;
            for (int pos = 0, x = -radius; x <= radius; ++x) {
                for (int y = -radius - std::min(0,x); y <= radius - std::max(0,x); ++y, ++pos) {
                    std::cout << std::setw(6) << updateCounterMatrix_[pos] << " ";
                }
                std::cout << std::endl;
            }
        } else {
            for (int pos = 0, d = 0; d != inputData_.som_depth; ++d) {
                for (int h = 0; h != inputData_.som_height; ++h) {
                    for (int w = 0; w != inputData_.som_width; ++w, ++pos) {
                        std::cout << std::setw(6) << updateCounterMatrix_[pos] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }
}

void SOM::updateSingleNeuron(float *neuron, float *image, float factor)
{
    for (int i = 0; i < inputData_.numberOfChannels * inputData_.neuron_size; ++i) {
        neuron[i] -= (neuron[i] - image[i]) * factor;
    }
}

} // namespace pink
