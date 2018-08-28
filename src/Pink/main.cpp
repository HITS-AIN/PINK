/**
 * @file   Pink/main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SOM.h"
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

using namespace pink;

int main (int argc, char **argv)
{
    #if DEBUG_MODE
        feenableexcept(FE_INVALID | FE_OVERFLOW);
    #endif

    // Start timer
    auto&& startTime = myclock::now();

    InputData inputData(argc, argv);
    SOM som(inputData);

    #if PINK_USE_CUDA
        if (inputData.useCuda)
        {
            if (inputData.useMultipleGPUs and cuda_getNumberOfGPUs() > 1)
                std::cout << "  Use multiple GPU code with " << cuda_getNumberOfGPUs() << " GPUs." << std::endl;
            else
                std::cout << "  Use single GPU code." << std::endl;

            if (inputData.executionPath == TRAIN)
                cuda_trainSelfOrganizingMap(inputData);
            else if (inputData.executionPath == MAP)
                cuda_mapping(inputData);
            else
                fatalError("Unknown execution path.");
        } else
    #endif
        if (inputData.executionPath == TRAIN)
            som.training();
        else if (inputData.executionPath == MAP)
            som.mapping();
        else
            fatalError("Unknown execution path.");

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
