/**
 * @file   Pink/main_generic.h
 * @brief  Generic main routine of PINK
 * @date   Oct 8, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "SelfOrganizingMapLib/SOM.h"
#include "SelfOrganizingMapLib/TrainerCPU.h"
#include "SelfOrganizingMapLib/TrainerGPU.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

template <typename SOMLayout, typename NeuronLayout, typename T>
void main_generic(InputData const & input_data)
{
	SOM<SOMLayout, NeuronLayout, T> som(input_data);

    auto&& distribution_function = GaussianFunctor(input_data.sigma, input_data.damping);

	if (input_data.executionPath == ExecutionPath::TRAIN)
	{
		if(input_data.useCuda) {
			TrainerGPU trainer(
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
	        	Cartesian<2, T> image({input_data.image_dim, input_data.image_dim},
	        	    iter_image_cur->getPointerOfFirstPixel());
	            trainer(som, image);
	        }
		} else {
			TrainerCPU trainer(
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
	        	Cartesian<2, T> image({input_data.image_dim, input_data.image_dim},
	        	    iter_image_cur->getPointerOfFirstPixel());
	            trainer(som, image);
	        }
		}
	} else if (input_data.executionPath == ExecutionPath::MAP) {
		//Mapper mapper;
	} else
    	pink::exception("Unknown execution path");
}

} // namespace pink
