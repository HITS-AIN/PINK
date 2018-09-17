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

TEST(SelfOrganizingMapTest, cartesian_2d)
{
	uint32_t som_size = 3;
	uint32_t image_size = 100;
	uint32_t neuron_size = image_size * std::sqrt(2.0) / 2.0;

	Cartesian<2, float> image({image_size, image_size}, 0.0);
	Cartesian<2, float> neuron({neuron_size, neuron_size}, 0.0);
	Cartesian<2, Cartesian<2, float>> som({som_size, som_size}, neuron);

	Trainer trainer;
	trainer(som, image);
}

TEST(SelfOrganizingMapTest, cartesian_3d)
{
	uint32_t som_size = 3;
	uint32_t image_size = 100;
	uint32_t neuron_size = image_size * std::sqrt(2.0) / 2.0;

	Cartesian<2, float> image({image_size, image_size}, 0.0);
	Cartesian<2, float> neuron({neuron_size, neuron_size}, 0.0);
	Cartesian<3, Cartesian<2, float>> som({som_size, som_size, som_size}, neuron);

	Trainer trainer;
	trainer(som, image);
}
