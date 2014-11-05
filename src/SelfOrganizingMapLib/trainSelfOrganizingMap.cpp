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
#include <iostream>
#include <iomanip>

using namespace std;
using namespace PINK;

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

	// Initialize SOM
	if (inputData.init == RANDOM) fillRandom(som, inputData.som_size * inputData.neuron_size, inputData.seed);
	else if (inputData.init == ZERO) fillZero(som, inputData.som_size * inputData.neuron_size);
    //writeSOM(som, som_dim, neuron_dim, "initial_som.bin");

	float progress = 0.0;
	float progressStep = 1.0 / inputData.numIter / inputData.numberOfImages;
	float nextProgressPrint = 0.0;

	for (int iter = 0; iter != inputData.numIter; ++iter)
	{
		int i = 0;
		for (ImageIterator<float> iterImage(inputData.imagesFilename),iterEnd; iterImage != iterEnd; ++i, ++iterImage)
		{
			if (inputData.verbose) {
				if (progress >= nextProgressPrint) {
					cout << "  Progress: " << fixed << setprecision(0) << progress*100 << " %" << endl;
					nextProgressPrint += inputData.progressFactor;
				}
				progress += progressStep;
			}

	//		stringstream ss;
	//		ss << "image" << i << ".bin";
	//		iterImage->writeBinary(ss.str());

			float *image = iterImage->getPointerOfFirstPixel();

			#if DEBUG_MODE
		        //checkArrayForNanAndNegative(image, image_size, "image");
			#endif

			for (int j = 0; j < inputData.image_size; ++j) {
				if (image[j] != image[j]) image[j] = 0.0;
			}

			generateRotatedImages(rotatedImages, image, inputData.numberOfRotations,
				inputData.image_dim, inputData.neuron_dim);

			#if DEBUG_MODE
		        checkArrayForNanAndNegative(rotatedImages, 2 * inputData.numberOfRotations * inputData.neuron_size, "rotatedImages");
		        checkArrayForNanAndNegative(som, inputData.som_size * inputData.neuron_size, "som");
			#endif

	//		stringstream ss2;
	//		ss2 << "rotatedImage" << i << ".bin";
	//		writeRotatedImages(rotatedImages, image_dim, numberOfRotations, ss2.str());

			generateEuclideanDistanceMatrix(euclideanDistanceMatrix, bestRotationMatrix,
				inputData.som_dim, som, inputData.neuron_dim, inputData.numberOfRotations, rotatedImages);

			Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, inputData.som_dim);

			//cout << "bestMatch = " << bestMatch << endl;

			updateNeurons(inputData.som_dim, som, inputData.neuron_dim, rotatedImages, bestMatch, bestRotationMatrix);

	//		stringstream ss3;
	//		ss3 << "som" << i << ".bin";
	//		writeSOM(som, som_dim, image_dim, ss3.str());
		}
	}

	free(rotatedImages);
	free(euclideanDistanceMatrix);
	free(bestRotationMatrix);

	#if DEBUG_MODE
	    checkArrayForNanAndNegative(som, inputData.som_size * inputData.neuron_size, "som");
	#endif

    if (inputData.verbose) {
	    cout << "  Progress: 100 %\n" << endl;
	    cout << "  Write final SOM to " << inputData.resultFilename << " ..." << endl;
    }

	writeSOM(som, inputData.som_dim, inputData.neuron_dim, inputData.resultFilename);
	free(som);
}
