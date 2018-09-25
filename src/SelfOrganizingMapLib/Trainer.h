/**
 * @file   SelfOrganizingMapLib/Trainer.h
 * @date   Sep 10, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

#include "ImageProcessingLib/CropAndRotate.h"

namespace pink {

class Trainer
{
public:

	Trainer(int verbosity = 0, int number_of_rotations = 360, bool use_flip = true,
		float progress_factor = 0.1, bool use_cuda = true)
     : verbosity(verbosity),
	   number_of_rotations(number_of_rotations),
	   use_flip(use_flip),
	   progress_factor(progress_factor),
	   use_cuda(use_cuda)
    {}

    template <typename SOMType>
	void operator () (SOMType& som, typename SOMType::value_type const& image) const
	{
//		auto&& list_of_spatial_transformed_images = SpatialTransformer(Rotation<0,1>(number_of_rotations), use_flip)(image);
//		auto&& [euclidean_distance] generate_euclidean_distance_matrix(som, list_of_spatial_transformed_images);
//
//		auto&& best_match = find_best_match();
//
//		update_counter(best_match);
//		update_neurons(best_match);
	}

private:

	int verbosity;
	int number_of_rotations;
	bool use_flip;
	float progress_factor;
	bool use_cuda;

};

} // namespace pink
