/**
 * @file   ImageProcessingLib/ImageRotator.h
 * @date   Sep 20, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <vector>

namespace pink {

struct ImageRotator
{
	template <typename ImageType>
	std::vector<ImageType> operator () (ImageType const& image) const
	{
	    std::vector<ImageType> rotated_images;



	    return rotated_images;
	}
};

} // namespace pink
