/**
 * @file   SelfOrganizingMapTest/training.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include "gtest/gtest.h"

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/SOM_generic.h"
#include "SelfOrganizingMapLib/Trainer.h"

using namespace pink;

TEST(SelfOrganizingMapTest, trainer_num_rot)
{
	EXPECT_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0,  -4, true, 0.1, false), std::runtime_error);
	EXPECT_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0,  -1, true, 0.1, false), std::runtime_error);
	EXPECT_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0,   0, true, 0.1, false), std::runtime_error);
	EXPECT_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0,  90, true, 0.1, false), std::runtime_error);

	EXPECT_NO_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0,   1, true, 0.1, false));
	EXPECT_NO_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0,   4, true, 0.1, false));
	EXPECT_NO_THROW(Trainer trainer(GaussianFunctor(1.1, 0.2), 0, 720, true, 0.1, false));
}

TEST(SelfOrganizingMapTest, trainer_cartesian_2d)
{
	typedef Cartesian<2, float> NeuronType;
	typedef SOM_generic<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;

	uint32_t som_size = 2;
	uint32_t image_size = 2;
	uint32_t neuron_size = 2;

	NeuronType image({image_size, image_size}, std::vector<float>{1., 1., 1., 1.});
	SOMType som({som_size, som_size}, {neuron_size, neuron_size}, 0.0);

	Trainer trainer(
	    GaussianFunctor(1.1, 0.2),  // std::function<float(float)> distribution_function
        0,                          // int verbosity
		4,                          // int number_of_rotations
		true,                       // bool use_flip
		0.1,                        // float progress_factor
        false,                      // bool use_cuda
		0                           // int max_update_distance
	);

	trainer(som, image);

	float v1 = GaussianFunctor(1.1, 0.2)(0.0);
	float v2 = GaussianFunctor(1.1, 0.2)(1.0);
	float v3 = GaussianFunctor(1.1, 0.2)(std::sqrt(2.0));

	for (int i = 0; i != 4; ++i) {
	    EXPECT_FLOAT_EQ(v1, som.get_neuron({0, 0}).get_data_pointer()[i]);
		EXPECT_FLOAT_EQ(v2, som.get_neuron({1, 0}).get_data_pointer()[i]);
		EXPECT_FLOAT_EQ(v2, som.get_neuron({0, 1}).get_data_pointer()[i]);
		EXPECT_FLOAT_EQ(v3, som.get_neuron({1, 1}).get_data_pointer()[i]);
	}
}
