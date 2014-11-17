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
#include "cublas_v2.h"
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <stdio.h>

using namespace std;
using namespace PINK;

void cuda_trainSelfOrganizingMap_algo0(InputData const& inputData)
{
    if (inputData.verbose) {
    	cout << "\n Starting CUDA version algorithm 0" << endl;
        cuda_print_properties();
    }

    cudaError_t error;

    float *som = NULL, *rotatedImages = NULL, *euclideanDistanceMatrix = NULL;
    float *d_som = NULL, *d_rotatedImages = NULL, *d_euclideanDistanceMatrix = NULL, *d_image = NULL;
    int *bestRotationMatrix = NULL;
    int *d_bestRotationMatrix = NULL;

	// Memory allocation
	if (inputData.verbose) cout << "\n  Size of SOM = " << inputData.som_total_size * sizeof(float) << " bytes" << endl;
	d_som = cuda_alloc_float(inputData.som_total_size);
	som = (float *)malloc(inputData.som_total_size * sizeof(float));

	if (inputData.verbose) cout << "  Size of rotated images = " << inputData.numberOfRotationsAndFlip * inputData.neuron_size * sizeof(float)<< " bytes" << endl;
	d_rotatedImages = cuda_alloc_float(inputData.numberOfRotationsAndFlip * inputData.neuron_size);
	rotatedImages = (float *)malloc(inputData.numberOfRotationsAndFlip * inputData.neuron_size * sizeof(float));

	if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << inputData.som_size * sizeof(float) << " bytes" << endl;
	d_euclideanDistanceMatrix = cuda_alloc_float(inputData.som_size);
	euclideanDistanceMatrix = (float *)malloc(inputData.som_size * sizeof(float));

	if (inputData.verbose) cout << "  Size of best rotation matrix = " << inputData.som_size * sizeof(int) << " bytes\n" << endl;
	d_bestRotationMatrix = cuda_alloc_int(inputData.som_size);
	bestRotationMatrix = (int *)malloc(inputData.som_size * sizeof(int));

	if (inputData.verbose) cout << "  Size of image = " << inputData.image_size * sizeof(float) << " bytes\n" << endl;
	d_image = cuda_alloc_float(inputData.image_size_using_flip);

    // Initialize SOM
	if (inputData.init == ZERO) cuda_fill_zero(d_som, inputData.som_total_size);
	else {
        printf("Random initialization not implemented yet.");
        exit(EXIT_FAILURE);
	}

    // Prepare trigonometric values
	float angleStepRadians = inputData.numberOfRotations ? 2.0 * M_PI / inputData.numberOfRotations : 0.0;

	float angle;
	float *cosAlpha = (float *)malloc(inputData.numberOfRotations * sizeof(float));
	float *d_cosAlpha = cuda_alloc_float(inputData.numberOfRotations);
	float *sinAlpha = (float *)malloc(inputData.numberOfRotations * sizeof(float));
	float *d_sinAlpha = cuda_alloc_float(inputData.numberOfRotations);

	for (int i = 0; i < inputData.numberOfRotations - 1; ++i) {
		angle = (i+1) * angleStepRadians;
	    cosAlpha[i] = cos(angle);
        sinAlpha[i] = sin(angle);
	}

	cuda_copyHostToDevice_float(d_cosAlpha, cosAlpha, inputData.numberOfRotations);
	cuda_copyHostToDevice_float(d_sinAlpha, sinAlpha, inputData.numberOfRotations);

	// Progress status
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

			generateRotatedImages(rotatedImages, iterImage->getPointerOfFirstPixel(), inputData.numberOfRotations,
				inputData.image_dim, inputData.neuron_dim, inputData.useFlip);

			cuda_copyHostToDevice_float(d_rotatedImages, rotatedImages, inputData.numberOfRotationsAndFlip * inputData.neuron_size);

			cuda_generateEuclideanDistanceMatrix_algo2(d_euclideanDistanceMatrix, d_bestRotationMatrix,
				inputData.som_dim, d_som, inputData.neuron_dim, inputData.numberOfRotationsAndFlip, d_rotatedImages);

			cuda_copyDeviceToHost_float(euclideanDistanceMatrix, d_euclideanDistanceMatrix, inputData.som_size);
			cuda_copyDeviceToHost_int(bestRotationMatrix, d_bestRotationMatrix, inputData.som_size);
			cuda_copyDeviceToHost_float(som, d_som, inputData.som_total_size);

			Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, inputData.som_dim);
			updateNeurons(inputData.som_dim, som, inputData.neuron_dim, rotatedImages, bestMatch, bestRotationMatrix);

			cuda_copyHostToDevice_float(d_som, som, inputData.som_total_size);
		}
	}

	if (inputData.verbose) {
		cout << "  Progress: 100 %\n" << endl;
		cout << "  Write final SOM to " << inputData.resultFilename << " ..." << endl;
	}

	writeSOM(som, inputData.som_dim, inputData.neuron_dim, inputData.resultFilename);

	// Free memory
	free(d_image);
	free(bestRotationMatrix);
	free(euclideanDistanceMatrix);
	free(rotatedImages);
	free(som);
	free(cosAlpha);
	free(sinAlpha);
	cuda_free(d_cosAlpha);
	cuda_free(d_sinAlpha);
    cuda_free(d_image);
    cuda_free(d_bestRotationMatrix);
    cuda_free(d_euclideanDistanceMatrix);
    cuda_free(d_rotatedImages);
    cuda_free(d_som);
}
