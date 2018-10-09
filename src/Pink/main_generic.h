/**
 * @file   Pink/main_generic.h
 * @brief  Generic main routine of PINK
 * @date   Oct 8, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/ImageIterator.h"
#include "SelfOrganizingMapLib/SOM_generic.h"
#include "SelfOrganizingMapLib/Trainer.h"

namespace pink {

template <typename SOMLayout, typename NeuronLayout, typename T>
void main_generic(InputData const & input_data)
{
	SOM_generic<SOMLayout, NeuronLayout, T> som;

    auto&& distribution_function = GaussianFunctor(input_data.sigma, input_data.damping);

	if (input_data.executionPath == ExecutionPath::TRAIN)
	{
        Trainer trainer(
            distribution_function,
            input_data.verbose,
			input_data.numberOfRotations,
			input_data.useFlip,
			input_data.progressFactor,
			input_data.useCuda,
			input_data.maxUpdateDistance
        );
        for (auto&& iter_image_cur = ImageIterator<T>(input_data.imagesFilename), iter_image_end = ImageIterator<T>();
        	iter_image_cur != iter_image_end; ++iter_image_cur)
        {
        	Cartesian<2, T> image({iter_image_cur->getWidth(),
                iter_image_cur->getHeight()}, iter_image_cur->getPointerOfFirstPixel());
            trainer(som, image);
        }
	} else if (input_data.executionPath == ExecutionPath::MAP) {
		//Mapper mapper;
	} else
    	std::runtime_error("Unknown execution path");
}

} // namespace pink
