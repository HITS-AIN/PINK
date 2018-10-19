/**
 * @file   Pink/main_generic.h
 * @brief  Generic main routine of PINK
 * @date   Oct 8, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <iostream>

#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/FileIO.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

#ifdef __CUDACC__
    #include "SelfOrganizingMapLib/TrainerGPU.h"
#else
    #include "SelfOrganizingMapLib/TrainerCPU.h"
#endif

namespace pink {

template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
void main_generic(InputData const & input_data)
{
    SOM<SOMLayout, DataLayout, T> som(input_data);

    auto&& distribution_function = GaussianFunctor(input_data.sigma, input_data.damping);

    if (input_data.executionPath == ExecutionPath::TRAIN)
    {
        Trainer<SOMLayout, DataLayout, T, UseGPU> trainer(
            som,
            distribution_function,
            input_data.verbose,
            input_data.numberOfRotations,
            input_data.useFlip,
            input_data.max_update_distance,
            input_data.interpolation
        );

        for (auto&& iter_image_cur = ImageIterator<T>(input_data.imagesFilename), iter_image_end = ImageIterator<T>();
            iter_image_cur != iter_image_end; ++iter_image_cur)
        {
        	auto&& beg = iter_image_cur->getPointerOfFirstPixel();
        	auto&& end = beg + iter_image_cur->getSize();
            Data<DataLayout, T> data({input_data.image_dim, input_data.image_dim}, std::vector<T>(beg, end));
            trainer(data);
        }

        std::cout << "  Write final SOM to " << input_data.resultFilename << " ... " << std::flush;
        write(som, input_data.resultFilename);
        std::cout << "done." << std::endl;
    }
    else if (input_data.executionPath == ExecutionPath::MAP)
    {
        //Mapper mapper;
    }
    else
    {
        pink::exception("Unknown execution path");
    }
}

} // namespace pink
