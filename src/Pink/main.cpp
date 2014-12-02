/**
 * @file   Pink/main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "UtilitiesLib/Error.h"
#include "UtilitiesLib/InputData.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <stdlib.h>

#if DEBUG_MODE
    #include <fenv.h>
#endif

#if PINK_USE_CUDA
    #include "CudaLib/CudaLib.h"
#endif

using namespace std;
using namespace PINK;
using namespace chrono;

int main (int argc, char **argv)
{
	#if DEBUG_MODE
		feenableexcept(FE_INVALID | FE_OVERFLOW);
	#endif

	// Start timer
	const auto startTime = steady_clock::now();

	InputData inputData(argc, argv);

    #if PINK_USE_CUDA
	    if (inputData.useCuda) {
	        if (inputData.executionPath == TRAIN)
                cuda_trainSelfOrganizingMap(inputData);
	        else if (inputData.executionPath == MAP)
	            cuda_mapping(inputData);
	        else
	            fatalError("Unknown execution path.");
		} else
    #endif
        if (inputData.executionPath == TRAIN)
	    	trainSelfOrganizingMap(inputData);
        else if (inputData.executionPath == MAP)
            mapping(inputData);
        else
            fatalError("Unknown execution path.");

	// Stop and print timer
	const auto stopTime = steady_clock::now();
	const auto duration = stopTime - startTime;
	cout << "\n  Total time (hh:mm:ss): "
		 << setfill('0') << setw(2) << duration_cast<hours>(duration).count() << ":"
		 << setfill('0') << setw(2) << duration_cast<minutes>(duration % hours(1)).count() << ":"
		 << setfill('0') << setw(2) << duration_cast<seconds>(duration % minutes(1)).count()
	     << "     (= " << duration_cast<seconds>(duration).count() << "s)" << endl;

    cout << "\n  Successfully finished. Have a nice day.\n" << endl;
	return 0;
}
