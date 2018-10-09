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

	if (input_data.executionPath == ExecutionPath::TRAIN)
	{
        Trainer trainer(
            GaussianFunctor(1.1, 0.2),
            input_data.verbose,
			input_data.numberOfRotations,
			input_data.useFlip,
			input_data.progressFactor,
			input_data.useCuda,
			input_data.maxUpdateDistance
        );
        for (auto&& iter_image_cur = ImageIterator<float>(input_data.imagesFilename), iter_image_end = ImageIterator<float>();
        	iter_image_cur != iter_image_end; ++iter_image_cur)
        {
        	Cartesian<2, float> image({3, 3}, iter_image_cur->getPointerOfFirstPixel());
            trainer(som, image);
        }
	} else if (input_data.executionPath == ExecutionPath::MAP) {
		//Mapper mapper;
	} else
    	std::runtime_error("Unknown execution path");
}

} // namespace pink
