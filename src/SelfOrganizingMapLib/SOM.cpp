/**
 * @file   SelfOrganizingMapLib/SOM.cpp
 * @brief  Self organizing Kohonen-map.
 * @date   Nov 25, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/Error.h"
#include "UtilitiesLib/Filler.h"
#include "SOM.h"

SOM::SOM(int mapDimension, int neuronDimension, int numberOfChannels, SOMInitialization initType,
    int seed, std::string const& filename)
 : mapDimension_(mapDimension), neuronDimension_(neuronDimension),
   mapSize_(mapDimension * mapDimension), neuronSize_(neuronDimension * neuronDimension),
   numberOfChannels_(numberOfChannels), data_(numberOfChannels * mapSize_ * neuronSize_)
{
    if (initType == ZERO)
        fillWithValue(&data_[0], data_.size());
    else if (initType == RANDOM)
        fillWithRandomNumbers(&data_[0], data_.size(), seed);
//	else if (initType == RANDOM_WITH_PREFERRED_DIRECTION) {
//	    fillWithRandomNumbers(&data_[0], data_.size(), seed);
//	    for (int i = 0; i < neuronDimension; ++i) data_[(0.5+i)*neuronDimension] = 1.0f;
//	}
    else if (initType == RANDOM_WITH_PREFERRED_DIRECTION) {
        fillWithRandomNumbers(&data_[0], data_.size(), seed);
        for (int n = 0; n < mapSize_; ++n)
            for (int c = 0; c < numberOfChannels_; ++c)
                for (int i = 0; i < neuronDimension_; ++i)
                    data_[(n*numberOfChannels_ + c)*neuronSize_ + i*neuronDimension_ + i] = 1.0f;
    }
    else if (initType == FILEINIT)
        readSOM(&data_[0], numberOfChannels_, mapDimension_, neuronDimension_, filename);
    else
        fatalError("Unknown initType.");
}

void SOM::write(std::string const& filename) const
{
	 writeSOM(&data_[0], numberOfChannels_, mapDimension_, neuronDimension_, filename);
}
