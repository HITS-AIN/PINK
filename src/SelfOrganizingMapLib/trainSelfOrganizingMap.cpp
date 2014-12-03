/**
 * @file   SelfOrganizingMapLib/trainSelfOrganizingMap.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "SOM.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/CheckArrays.h"
#include "UtilitiesLib/Filler.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;
using namespace PINK;
using namespace chrono;

void trainSelfOrganizingMap(InputData const& inputData)
{
    if (inputData.verbose) cout << "  Starting C version of training.\n" << endl;

	// Memory allocation
	int rotatedImagesSize = inputData.numberOfChannels * inputData.numberOfRotations * inputData.neuron_size;
	if (inputData.useFlip) rotatedImagesSize *= 2;
	if (inputData.verbose) cout << "\n  Size of rotated images = " << rotatedImagesSize * sizeof(float) << " bytes" << endl;
	vector<float> rotatedImages(rotatedImagesSize);

	if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << inputData.som_size * sizeof(float) << " bytes" << endl;
	vector<float> euclideanDistanceMatrix(inputData.som_size);

	if (inputData.verbose) cout << "  Size of best rotation matrix = " << inputData.som_size * sizeof(int) << " bytes" << endl;
	vector<int> bestRotationMatrix(inputData.som_size);

	// Initialize SOM
	SOM som(inputData.som_dim, inputData.neuron_dim, inputData.numberOfChannels, inputData.init, inputData.seed, inputData.somFilename);
	if (inputData.verbose) cout << "  Size of SOM = " << som.getSizeInBytes() << " bytes\n" << endl;
    //som.write("initial_som.bin");

    // Counting updates of each neuron
    vector<int> updateCounter(inputData.som_size);

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

                if (inputData.intermediate_storage) {
                    if (inputData.verbose) cout << "  Write intermediate SOM to " << inputData.resultFilename << " ... " << flush;
                    som.write(inputData.resultFilename);
                    if (inputData.verbose) cout << "done." << endl;
                }

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

            Point bestMatch = findBestMatchingNeuron(&euclideanDistanceMatrix[0], inputData.som_dim);
            ++updateCounter[bestMatch.x*inputData.som_dim + bestMatch.y];

            updateNeurons(inputData.som_dim, som.getDataPointer(), inputData.neuron_dim, &rotatedImages[0],
                bestMatch, &bestRotationMatrix[0], inputData.numberOfChannels);
		}
	}

	#if DEBUG_MODE
	    checkArrayForNan(som.getDataPointer(), inputData.som_size * inputData.neuron_size, "som");
	#endif

	cout << "  Progress: 100 % ("
		 << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;
	cout << "  Write final SOM to " << inputData.resultFilename << " ... " << flush;

	som.write(inputData.resultFilename);
	cout << "done." << endl;

	if (inputData.verbose) {
        cout << "\n  Number of updates of each neuron:\n" << endl;
        for (int i=0; i != inputData.som_dim; ++i) {
            for (int j=0; j != inputData.som_dim; ++j) {
                cout << setw(6) << updateCounter[i*inputData.som_dim + j] << " ";
            }
            cout << endl;
        }
	}
}
