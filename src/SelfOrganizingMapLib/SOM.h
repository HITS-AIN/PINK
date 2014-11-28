/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @brief  Self organizing Kohonen-map.
 * @date   Nov 25, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef SOM_H_
#define SOM_H_

#include "UtilitiesLib/InputData.h"
#include <string>
#include <vector>

class SOM
{
public:

	SOM(int mapDimension, int neuronDimension, int numberOfChannels, SOMInitialization initType,
	    int seed, std::string const& filename);

	void write(std::string const& filename) const;

	int getSize() const { return data_.size(); }

    int getSizeInBytes() const { return data_.size() * sizeof(float); }

	std::vector<float> getData() { return data_; }

	const std::vector<float> getData() const { return data_; }

    float* getDataPointer() { return &data_[0]; }

    float const* getDataPointer() const { return &data_[0]; }

private:

    int mapDimension_;
    int mapSize_;
    int neuronDimension_;
    int neuronSize_;
    int numberOfChannels_;

	std::vector<float> data_;

};

#endif /* SOM_H_ */
