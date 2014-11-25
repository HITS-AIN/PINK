/**
 * @file   trainSelfOrganizingMap.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/CheckArrays.h"
#include "UtilitiesLib/Filler.h"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace PINK;
using namespace chrono;

void trainSelfOrganizingMap(InputData const& inputData)
{
	// Memory allocation
	int som_sizeInBytes = inputData.som_size * inputData.neuron_size * sizeof(float);
	if (inputData.verbose) cout << "\n  Size of SOM = " << som_sizeInBytes << " bytes" << endl;
	float *som = (float *)malloc(som_sizeInBytes);

	int rotatedImages_sizeInBytes = 2 * inputData.numberOfRotations * inputData.neuron_size * sizeof(float);
	if (inputData.verbose) cout << "  Size of rotated images = " << rotatedImages_sizeInBytes << " bytes" << endl;
	float *rotatedImages = (float *)malloc(rotatedImages_sizeInBytes);

	int euclideanDistanceMatrix_sizeInBytes = inputData.som_size * sizeof(float);
	if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << euclideanDistanceMatrix_sizeInBytes << " bytes" << endl;
	float *euclideanDistanceMatrix = (float *)malloc(euclideanDistanceMatrix_sizeInBytes);

	int bestRotationMatrix_sizeInBytes = inputData.som_size * sizeof(int);
	if (inputData.verbose) cout << "  Size of best rotation matrix = " << bestRotationMatrix_sizeInBytes << " bytes\n" << endl;
	int *bestRotationMatrix = (int *)malloc(bestRotationMatrix_sizeInBytes);

    // Counting updates of each neuron
	vector<int> updateCounter(inputData.som_size);

	// Initialize SOM
	if (inputData.init == RANDOM) fillWithRandomNumbers(som, inputData.som_size * inputData.neuron_size, inputData.seed);
	else if (inputData.init == ZERO) fillWithValue(som, inputData.som_size * inputData.neuron_size);
    //writeSOM(som, som_dim, neuron_dim, "initial_som.bin");

	float progress = 0.0;
	float progressStep = 1.0 / inputData.numIter / inputData.numberOfImages;
	float nextProgressPrint = inputData.progressFactor;

	// Start timer
	auto startTime = steady_clock::now();

	for (int iter = 0; iter != inputData.numIter; ++iter)
	{
		int i = 0;
		for (ImageIterator<float> iterImage(inputData.imagesFilename),iterEnd; iterImage != iterEnd; ++i, iterImage += iterImage.getNumberOfChannels())
		{
			if (progress >= nextProgressPrint)
			{
				const auto stopTime = steady_clock::now();
				const auto duration = stopTime - startTime;

				cout << "  Progress: " << fixed << setprecision(0) << progress*100 << " % ("
					 << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;
				cout << "  Write intermediate SOM to " << inputData.resultFilename << " ... " << flush;

				writeSOM(som, inputData.som_dim, inputData.neuron_dim, inputData.resultFilename);
				cout << "done." << endl;

				nextProgressPrint += inputData.progressFactor;
				startTime = steady_clock::now();
			}
			progress += progressStep;

			float *image = iterImage->getPointerOfFirstPixel();

			#if DEBUG_MODE
		        checkArrayForNan(image, inputData.image_size, "image");
			#endif

			generateRotatedImages(rotatedImages, image, inputData.numberOfRotations,
				inputData.image_dim, inputData.neuron_dim, inputData.useFlip, inputData.interpolation);

			#if DEBUG_MODE
		        checkArrayForNan(rotatedImages, inputData.numberOfRotationsAndFlip * inputData.neuron_size, "rotatedImages");
		        checkArrayForNan(som, inputData.som_size * inputData.neuron_size, "som");
			#endif

			generateEuclideanDistanceMatrix(euclideanDistanceMatrix, bestRotationMatrix,
				inputData.som_dim, som, inputData.neuron_dim, inputData.numberOfRotationsAndFlip, rotatedImages);

			Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, inputData.som_dim);
			++updateCounter[bestMatch.x*inputData.som_dim + bestMatch.y];

			updateNeurons(inputData.som_dim, som, inputData.neuron_dim, rotatedImages, bestMatch, bestRotationMatrix);
		}
	}

	free(rotatedImages);
	free(euclideanDistanceMatrix);
	free(bestRotationMatrix);

	#if DEBUG_MODE
	    checkArrayForNan(som, inputData.som_size * inputData.neuron_size, "som");
	#endif

	cout << "  Progress: 100 % ("
		 << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;
	cout << "  Write final SOM to " << inputData.resultFilename << " ... " << flush;

	writeSOM(som, inputData.som_dim, inputData.neuron_dim, inputData.resultFilename);
	cout << "done." << endl;

	cout << "\n  Number of updates of each neuron:\n" << endl;
	for (int i=0; i != inputData.som_dim; ++i) {
		for (int j=0; j != inputData.som_dim; ++j) {
			cout << setw(6) << updateCounter[i*inputData.som_dim + j] << " ";
		}
		cout << endl;
	}

	free(som);
}
