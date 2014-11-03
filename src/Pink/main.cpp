/**
 * @file   Pink/main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "UtilitiesLib/InputData.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#if DEBUG_MODE
    #include <fenv.h>
#endif

#if PINK_USE_CUDA
    #include "CudaLib/CudaLib.h"
#endif

using namespace std;
using namespace PINK;
using namespace chrono;

void print_header()
{
	cout << "\n"
	        "  ************************************************************************\n"
	        "  *   Parallel orientation Invariant Non-parametric Kohonen-map (PINK)   *\n"
	        "  ************************************************************************\n" << endl;
}

int main (int argc, char **argv)
{
	#if DEBUG_MODE
		feenableexcept(FE_INVALID | FE_OVERFLOW);
	#endif

	// Start timer
	const auto startTime = steady_clock::now();

	print_header();

	InputData inputData(argc, argv);

	if (inputData.numberOfThreads == -1) inputData.numberOfThreads = omp_get_num_procs();
	omp_set_num_threads(inputData.numberOfThreads);

	inputData.print();

    #if PINK_USE_CUDA
	    if (inputData.useCuda)
	    	cuda_trainSelfOrganizingMap(inputData);
	    else
    #endif
	    	trainSelfOrganizingMap(inputData);

	// Stop and print timer
	const auto stopTime = steady_clock::now();
	const auto duration = stopTime - startTime;
	cout << "\n  Total time (hh:mm:ss): "
		 << setfill('0') << setw(2) << duration_cast<hours>(duration).count() << ":"
		 << setfill('0') << setw(2) << duration_cast<minutes>(duration % hours(1)).count() << ":"
		 << setfill('0') << setw(2) << duration_cast<seconds>(duration % minutes(1)).count() << endl;

    cout << "\n  Successfully finished. Have a nice day.\n" << endl;
	return 0;
}
