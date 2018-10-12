/**
 * @file   Pink/main_generic.h
 * @brief  Generic main routine of PINK
 * @date   Oct 8, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "SelfOrganizingMapLib/TrainerCPU.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T, bool use_gpu>
void main_generic(InputData const & input_data)
{
	SOM<SOMLayout, DataLayout, T> som(input_data);

    auto&& distribution_function = GaussianFunctor(input_data.sigma, input_data.damping);

	if (input_data.executionPath == ExecutionPath::TRAIN)
	{
		Trainer<SOMLayout, DataLayout, T, use_gpu> trainer(
			distribution_function,
			input_data.verbose,
			input_data.numberOfRotations,
			input_data.useFlip,
			input_data.progressFactor,
			input_data.maxUpdateDistance
		);

		for (auto&& iter_image_cur = ImageIterator<T>(input_data.imagesFilename), iter_image_end = ImageIterator<T>();
			iter_image_cur != iter_image_end; ++iter_image_cur)
		{
			Data<DataLayout, T> data({input_data.image_dim, input_data.image_dim},
				iter_image_cur->getPointerOfFirstPixel());
			trainer(som, data);
		}
	} else if (input_data.executionPath == ExecutionPath::MAP) {
		//Mapper mapper;
	} else
    	pink::exception("Unknown execution path");
}

} // namespace pink
