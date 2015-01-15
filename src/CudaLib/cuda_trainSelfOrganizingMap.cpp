/**
 * @file   CudaLib/cuda_trainSelfOrganizingMap.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/Filler.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace std;
using namespace PINK;
using namespace chrono;

void cuda_trainSelfOrganizingMap(InputData const& inputData)
{
    cout << "  Starting CUDA version of training.\n" << endl;
    if (inputData.verbose) cuda_print_properties();

    cuda_setDevice(0);

    // Initialize SOM on host
    SOM som(inputData);
    if (inputData.verbose) cout << "\n  Size of SOM = " << som.getSize() * sizeof(float) << " bytes" << endl;
    float *d_som = cuda_alloc_float(som.getSize());
    cuda_copyHostToDevice_float(d_som, som.getDataPointer(), som.getSize());

    // Memory allocation
    int rotatedImagesSize = inputData.numberOfChannels * inputData.numberOfRotationsAndFlip * inputData.neuron_size;
    if (inputData.verbose) cout << "  Size of rotated images = " << rotatedImagesSize * sizeof(float)<< " bytes" << endl;
    float *d_rotatedImages = cuda_alloc_float(rotatedImagesSize);

    if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << inputData.som_size * sizeof(float) << " bytes" << endl;
    float *d_euclideanDistanceMatrix = cuda_alloc_float(inputData.som_size);

    if (inputData.verbose) cout << "  Size of best rotation matrix = " << inputData.som_size * sizeof(int) << " bytes\n" << endl;
    int *d_bestRotationMatrix = cuda_alloc_int(inputData.som_size);

    if (inputData.verbose) cout << "  Size of image = " << inputData.numberOfChannels * inputData.image_size * sizeof(float) << " bytes\n" << endl;
    float *d_image = cuda_alloc_float(inputData.numberOfChannels * inputData.image_size);

    // Best matching position (x,y)
    int bestMatch;
    int *d_bestMatch = cuda_alloc_int(1);

    // Prepare trigonometric values
    float *d_cosAlpha = NULL, *d_sinAlpha = NULL;
    trigonometricValues(&d_cosAlpha, &d_sinAlpha, inputData.numberOfRotations/4);

    // Progress status
    float progress = 0.0;
    float progressStep = 1.0 / inputData.numIter / inputData.numberOfImages;
    float nextProgressPrint = inputData.progressFactor;
    int progressPrecision = rint(log10(1.0 / inputData.progressFactor)) - 2;
    if (progressPrecision < 0) progressPrecision = 0;

    int interStoreCount = 0;
    int updateCount = 0;

    // Start timer
    auto startTime = steady_clock::now();

    for (int iter = 0; iter != inputData.numIter; ++iter)
    {
        for (ImageIterator<float> iterImage(inputData.imagesFilename),iterEnd; iterImage != iterEnd; ++iterImage, ++updateCount)
        {
            if ((inputData.progressFactor < 1.0 and progress > nextProgressPrint) or
                (inputData.progressFactor >= 1.0 and updateCount != 0 and !(updateCount % static_cast<int>(inputData.progressFactor))))
            {
                cout << "  Progress: " << setw(12) << updateCount << " updates, "
                     << fixed << setprecision(progressPrecision) << setw(3) << progress*100 << " % ("
                     << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;

                if (inputData.intermediate_storage != OFF) {
                    string interStoreFilename = inputData.resultFilename;
                    if (inputData.intermediate_storage == KEEP) {
                        interStoreFilename.insert(interStoreFilename.find_last_of("."), "_" + to_string(interStoreCount));
                        ++interStoreCount;
                    }
                    if (inputData.verbose) cout << "  Write intermediate SOM to " << interStoreFilename << " ... " << flush;
                    cuda_copyDeviceToHost_float(som.getDataPointer(), d_som, som.getSize());
                    som.write(interStoreFilename);
                    if (inputData.verbose) cout << "done." << endl;
                }

                nextProgressPrint += inputData.progressFactor;
                startTime = steady_clock::now();
            }
            progress += progressStep;

            cuda_copyHostToDevice_float(d_image, iterImage->getPointerOfFirstPixel(), iterImage->getSize());

            cuda_generateRotatedImages(d_rotatedImages, d_image, inputData.numberOfRotations,
                inputData.image_dim, inputData.neuron_dim, inputData.useFlip, inputData.interpolation,
                d_cosAlpha, d_sinAlpha, inputData.numberOfChannels);

            cuda_generateEuclideanDistanceMatrix(d_euclideanDistanceMatrix, d_bestRotationMatrix,
                inputData.som_size, d_som, inputData.numberOfChannels * inputData.neuron_size,
                inputData.numberOfRotationsAndFlip, d_rotatedImages, inputData.block_size_1, inputData.useMultipleGPUs);

            cuda_updateNeurons(d_som, d_rotatedImages, d_bestRotationMatrix, d_euclideanDistanceMatrix, d_bestMatch,
                inputData.som_width, inputData.som_height, inputData.som_depth, inputData.som_size,
                inputData.numberOfChannels * inputData.neuron_size, inputData.numberOfRotationsAndFlip,
                inputData.function, inputData.layout, inputData.sigma, inputData.damping, inputData.maxUpdateDistance,
                inputData.usePBC, inputData.dimensionality);

            cuda_copyDeviceToHost_int(&bestMatch, d_bestMatch, 1);
            som.updateCounter(bestMatch);
        }
    }

    // Free memory
    if (d_cosAlpha) cuda_free(d_cosAlpha);
    if (d_sinAlpha) cuda_free(d_sinAlpha);
    cuda_free(d_image);
    cuda_free(d_bestRotationMatrix);
    cuda_free(d_euclideanDistanceMatrix);
    cuda_free(d_rotatedImages);
    cuda_free(d_bestMatch);

    cout << "  Progress: " << setw(12) << updateCount << " updates, 100 % ("
         << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;

    cout << "  Write final SOM to " << inputData.resultFilename << " ... " << flush;
    cuda_copyDeviceToHost_float(som.getDataPointer(), d_som, som.getSize());
    som.write(inputData.resultFilename);
    cout << "done." << endl;

    som.printUpdateCounter();

    // Free memory
    cuda_free(d_som);
}
