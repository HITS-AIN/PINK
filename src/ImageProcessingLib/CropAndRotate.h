/**
 * @file   ImageProcessingLib/CropAndRotate.h
 * @date   Sep 20, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cmath>
#include <vector>

namespace pink {

/// The images will be rotated and cropped to remove the rotation edge.
/// The image size of the rotated images will the input image size times std::sqrt(2.0) / 2.0.
/// The image must be quadratic.
struct CropAndRotate
{
	CropAndRotate(uint32_t number_of_rotations)
	 : number_of_rotations(number_of_rotations)
	{
		assert(number_of_rotations % 4 == 0);
	}

	template <typename ImageType>
	std::vector<ImageType> operator () (ImageType const& image) const
	{
		/// Check if image is quadratic
		assert(image.get_lenght()[0] == image.get_lenght()[1]);

		auto&& image_dim = image.get_length()[0];
		int cropped_image_dim = image_dim * std::sqrt(2.0) / 2.0;

	    std::vector<ImageType> rotated_images(number_of_rotations);

		int num_real_rot = number_of_rotations/4;
		float angleStepRadians = 2.0 * M_PI / number_of_rotations;

		// Copy original image to first position of image array
		rotated_images[0] = crop(image);
		rotated_images[num_real_rot] = rotate_90degrees(rotated_images[0]);
		rotated_images[2 * num_real_rot] = rotate_90degrees(rotated_images[num_real_rot]);
		rotated_images[3 * num_real_rot] = rotate_90degrees(rotated_images[2 * num_real_rot]);

		// Rotate images
		#pragma omp parallel for
		for (int i = 1; i < num_real_rot; ++i) {
			rotated_images[i] = rotate_and_crop(image, i*angleStepRadians, interpolation);
			rotated_images[i + num_real_rot] = rotate_90degrees(rotated_images[i]);
			rotated_images[i + 2 * num_real_rot] = rotate_90degrees(rotated_images[i + num_real_rot]);
			rotated_images[i + 3 * num_real_rot] = rotate_90degrees(rotated_images[i + 2 * num_real_rot]);
		}

	    return rotated_images;
	}

private:

	uint32_t number_of_rotations;

};

} // namespace pink
