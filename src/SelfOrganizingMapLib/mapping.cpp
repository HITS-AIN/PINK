/**
 * @file   SelfOrganizingMapLib/mapping.cpp
 * @date   Nov 28, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/ImageIterator.h"
#include "SelfOrganizingMap.h"
#include "SOM.h"
#include "UtilitiesLib/Error.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace PINK;
using namespace chrono;

void mapping(InputData const& inputData)
{
    if (inputData.verbose) cout << "  Starting C version of mapping.\n" << endl;

    // Open result file
    std::ofstream resultFile(inputData.resultFilename);
    if (!resultFile) fatalError("Error opening " + inputData.resultFilename);
    resultFile.write((char*)&inputData.numberOfImages, sizeof(int));
    resultFile.write((char*)&inputData.som_dim, sizeof(int));
    resultFile.write((char*)&inputData.som_dim, sizeof(int));

    // Initialize SOM on host
    SOM som(inputData.som_dim, inputData.neuron_dim, inputData.numberOfChannels,
        FILEINIT, inputData.seed, inputData.somFilename);

    // Memory allocation
    int rotatedImagesSize = inputData.numberOfChannels * inputData.numberOfRotations * inputData.neuron_size;
    if (inputData.useFlip) rotatedImagesSize *= 2;
    if (inputData.verbose) cout << "\n  Size of rotated images = " << rotatedImagesSize * sizeof(float) << " bytes" << endl;
    vector<float> rotatedImages(rotatedImagesSize);

    if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << inputData.som_size * sizeof(float) << " bytes" << endl;
    vector<float> euclideanDistanceMatrix(inputData.som_size);

    if (inputData.verbose) cout << "  Size of best rotation matrix = " << inputData.som_size * sizeof(int) << " bytes\n" << endl;
    vector<int> bestRotationMatrix(inputData.som_size);

    float progress = 0.0;
    float progressStep = 1.0 / inputData.numIter / inputData.numberOfImages;
    float nextProgressPrint = inputData.progressFactor;

    // Start timer
    auto startTime = steady_clock::now();

    for (int iter = 0; iter != inputData.numIter; ++iter)
    {
        for (ImageIterator<float> iterImage(inputData.imagesFilename), iterEnd; iterImage != iterEnd; ++iterImage)
        {
            if (progress >= nextProgressPrint)
            {
                const auto stopTime = steady_clock::now();
                const auto duration = stopTime - startTime;

                cout << "  Progress: " << fixed << setprecision(0) << progress*100 << " % ("
                     << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;

                nextProgressPrint += inputData.progressFactor;
                startTime = steady_clock::now();
            }
            progress += progressStep;

            generateRotatedImages(&rotatedImages[0], iterImage->getPointerOfFirstPixel(), inputData.numberOfRotations,
                inputData.image_dim, inputData.neuron_dim, inputData.useFlip, inputData.interpolation,
                inputData.numberOfChannels);

            generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
                inputData.som_dim, som.getDataPointer(), inputData.neuron_dim, inputData.numberOfRotationsAndFlip,
                &rotatedImages[0], inputData.numberOfChannels);

            resultFile.write((char*)&euclideanDistanceMatrix[0], inputData.som_size * sizeof(float));
        }
    }

    cout << "  Progress: 100 % ("
         << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;
}
