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

#include "main_cpu.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

#if PINK_USE_CUDA
    #include "CudaLib/main_gpu.h"
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

		if (input_data.use_gpu)
#if PINK_USE_CUDA
            main_gpu(input_data);
#else
		    pink::exception("PINK was not compiled with CUDA support");
#endif
        else
            main_cpu(input_data);

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
