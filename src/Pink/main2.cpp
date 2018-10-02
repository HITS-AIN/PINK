/**
 * @file   Pink/main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

#ifndef NDEBUG
    #include <fenv.h>
#endif

#include "ImageProcessingLib/ImageIterator.h"
#include "SelfOrganizingMapLib/SOM_generic.h"
#include "SelfOrganizingMapLib/Trainer.h"
#include "UtilitiesLib/InputData.h"

#if PINK_USE_CUDA
    #include "CudaLib/CudaLib.h"
#endif

using myclock = std::chrono::steady_clock;

using namespace pink;

int main(int argc, char **argv)
{
    #ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
    #endif

    // Start timer
    auto&& startTime = myclock::now();

    InputData input_data(argc, argv);

    if (input_data.layaut == Layout::CARTESIAN)
        if (input_data.dimensionality == 1)
        	main_generic<CartesianLayout<1>, CartesianLayout<2>, float>();
        else if (input_data.dimensionality == 2)
			main_generic<CartesianLayout<2>, CartesianLayout<2>, float>();
        else if (input_data.dimensionality == 3)
			main_generic<CartesianLayout<3>, CartesianLayout<2>, float>();
        else
        	std::runtime_error("Unsupported dimensionality of " + input_data.dimensionality);
    else if (input_data.layaut == Layout::HEXAGONAL)
    	main_generic<HexagonalLayout, CartesianLayout<2>, float>();
    else
    	std::runtime_error("Unknown layout");

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

    // Stop and print timer
    auto&& stopTime = myclock::now();
    auto&& duration = stopTime - startTime;
    std::cout << "\n  Total time (hh:mm:ss): "
         << std::setfill('0') << std::setw(2) << std::chrono::duration_cast<std::chrono::hours>(duration).count() << ":"
         << std::setfill('0') << std::setw(2) << std::chrono::duration_cast<std::chrono::minutes>(duration % std::chrono::hours(1)).count() << ":"
         << std::setfill('0') << std::setw(2) << std::chrono::duration_cast<std::chrono::seconds>(duration % std::chrono::minutes(1)).count()
         << "     (= " << std::chrono::duration_cast<std::chrono::seconds>(duration).count() << "s)" << std::endl;

    std::cout << "\n  Successfully finished. Have a nice day.\n" << std::endl;
    return 0;
}
