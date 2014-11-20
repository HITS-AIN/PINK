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
	    if (inputData.useCuda)
	    {
			switch (inputData.algo)
			{
			    case 0:
				{
			    	cuda_trainSelfOrganizingMap_algo0(inputData);
					break;
			    }
			    case 1:
				{
			    	cuda_trainSelfOrganizingMap_algo1(inputData);
					break;
			    }
			    case 2:
				{
			    	cuda_trainSelfOrganizingMap_algo2(inputData);
					break;
			    }
			    default:
				{
			    	cout << "Unkown algorithm number (" << inputData.algo << ")." << endl;
				    exit(EXIT_FAILURE);
			    }
			}
		} else
    #endif
	    	trainSelfOrganizingMap(inputData);

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
