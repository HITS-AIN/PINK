/**
 * @file   Pink/main2.cpp
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

#include "main_generic.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

#if PINK_USE_CUDA
    #include "CudaLib/CudaLib.h"
#endif

using myclock = std::chrono::steady_clock;
using namespace pink;

int main(int argc, char **argv)
{
    try {
		#ifndef NDEBUG
			feenableexcept(FE_INVALID | FE_OVERFLOW);
		#endif

		// Start timer
		auto&& startTime = myclock::now();

		InputData input_data(argc, argv);

		if (input_data.layout == Layout::CARTESIAN)
			if (input_data.dimensionality == 1)
				main_generic<CartesianLayout<1>, CartesianLayout<2>, float, false>(input_data);
			else if (input_data.dimensionality == 2)
				main_generic<CartesianLayout<2>, CartesianLayout<2>, float, false>(input_data);
			else if (input_data.dimensionality == 3)
				main_generic<CartesianLayout<3>, CartesianLayout<2>, float, false>(input_data);
			else
				pink::exception("Unsupported dimensionality of " + input_data.dimensionality);
		else if (input_data.layout == Layout::HEXAGONAL)
			main_generic<HexagonalLayout, CartesianLayout<2>, float, false>(input_data);
		else
			pink::exception("Unknown layout");

		// Stop and print timer
		auto&& stopTime = myclock::now();
		auto&& duration = stopTime - startTime;
		std::cout << "\n  Total time (hh:mm:ss): "
			 << std::setfill('0') << std::setw(2) << std::chrono::duration_cast<std::chrono::hours>(duration).count() << ":"
			 << std::setfill('0') << std::setw(2) << std::chrono::duration_cast<std::chrono::minutes>(duration % std::chrono::hours(1)).count() << ":"
			 << std::setfill('0') << std::setw(2) << std::chrono::duration_cast<std::chrono::seconds>(duration % std::chrono::minutes(1)).count()
			 << "     (= " << std::chrono::duration_cast<std::chrono::seconds>(duration).count() << "s)" << std::endl;


    } catch ( pink::exception const& e ) {
        std::cout << "PINK exception: " << e.what() << std::endl;
        std::cout << "Program was aborted." << std::endl;
        return 1;
    } catch ( std::exception const& e ) {
        std::cout << "Standard exception: " << e.what() << std::endl;
        std::cout << "Program was aborted." << std::endl;
        return 1;
    } catch ( ... ) {
        std::cout << "Unknown exception." << std::endl;
        std::cout << "Program was aborted." << std::endl;
        return 1;
    }

    std::cout << "\n  Successfully finished. Have a nice day.\n" << std::endl;
    return 0;
}
