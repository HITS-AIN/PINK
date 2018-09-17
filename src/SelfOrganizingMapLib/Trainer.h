/**
 * @file   SelfOrganizingMapLib/train.h
 * @date   Sep 10, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

struct Trainer
{
	template <typename SOMType, typename ImageType>
	void operator () (SOMType& som, ImageType const& image) const
	{
//		auto&& rotated_images = generate_rotated_images(image);
//		generate_euclidean_distance_matrix();
//
//		auto&& best_match = find_best_match();
//
//		update_counter(best_match);
//		update_neurons(best_match);
	}
};

} // namespace pink
