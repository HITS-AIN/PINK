/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @brief  Self organizing Kohonen-map.
 * @date   Nov 25, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "UtilitiesLib/DistanceFunctor.h"
#include "UtilitiesLib/DistributionFunctor.h"
#include "UtilitiesLib/InputData.h"

using myclock = std::chrono::steady_clock;

namespace pink {

/**
 * @brief Main class for self organizing matrix.
 */
class SOM
{
public:

    SOM(InputData const& inputData);

    void write(std::string const& filename) const;

    int getSize() const { return som_.size(); }

    int getSizeInBytes() const { return som_.size() * sizeof(float); }

    std::vector<float> getData() { return som_; }

    const std::vector<float> getData() const { return som_; }

    float* getDataPointer() { return &som_[0]; }

    float const* getDataPointer() const { return &som_[0]; }

    //! Main CPU based routine for SOM training.
    void training();

    //! Main CPU based routine for SOM mapping.
    void mapping();

    //! Updating self organizing map.
    void updateNeurons(float *rotatedImages, int bestMatch, int *bestRotationMatrix);

    //! Save position of current SOM update.
    void updateCounter(int bestMatch) { ++updateCounterMatrix_[bestMatch]; }

    //! Print matrix of SOM updates.
    void printUpdateCounter() const;

private:

    //! Updating one single neuron.
    void updateSingleNeuron(float *neuron, float *image, float factor);

    InputData const& inputData_;

    //! The real self organizing matrix.
    std::vector<float> som_;

    std::shared_ptr<DistributionFunctorBase> ptrDistributionFunctor_;

    std::shared_ptr<DistanceFunctorBase> ptrDistanceFunctor_;

    // Counting updates of each neuron
    std::vector<int> updateCounterMatrix_;

};

} // namespace pink
