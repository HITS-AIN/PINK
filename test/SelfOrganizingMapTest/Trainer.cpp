/**
 * @file   SelfOrganizingMapTest/training.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Trainer.h"

using namespace pink;

TEST(SelfOrganizingMapTest, train_cartesian_2d)
{
	typedef Cartesian<2, float> NeuronType;
	typedef Cartesian<2, NeuronType> SOMType;

	uint32_t som_size = 3;
	uint32_t image_size = 100;
	uint32_t neuron_size = image_size * std::sqrt(2.0) / 2.0;

	NeuronType image({image_size, image_size});
	NeuronType neuron({neuron_size, neuron_size});
	SOMType som({som_size, som_size}, std::vector<NeuronType>(som_size * som_size, neuron));

	Trainer trainer;
	trainer(som, image);
}

TEST(SelfOrganizingMapTest, train_cartesian_3d)
{
	typedef Cartesian<2, float> NeuronType;
	typedef Cartesian<3, NeuronType> SOMType;

	uint32_t som_size = 3;
	uint32_t image_size = 100;
	uint32_t neuron_size = image_size * std::sqrt(2.0) / 2.0;

	NeuronType image({image_size, image_size});
	NeuronType neuron({neuron_size, neuron_size});
	SOMType som({som_size, som_size}, std::vector<NeuronType>(som_size * som_size, neuron));

	Trainer trainer;
	trainer(som, image);
}
