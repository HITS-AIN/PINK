/**
 * @file   cuda_trainSelfOrganizingMap.cu
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

void cuda_trainSelfOrganizingMap(InputData const& inputData)
{
    if (inputData.verbose) {
    	cuda_print_properties();
		std::cout << "  Number of CPU threads is reset to one using CUDA." << std::endl;
	}
    omp_set_num_threads(1);

    cudaError_t error;

    float *som = NULL, *rotatedImages = NULL, *euclideanDistanceMatrix = NULL;
    float *d_som = NULL, *d_rotatedImages = NULL, *d_euclideanDistanceMatrix = NULL, *d_image = NULL;
    int *bestRotationMatrix = NULL;
    int *d_bestRotationMatrix = NULL;

	// Memory allocation
	if (inputData.verbose) cout << "\n  Size of SOM = " << inputData.som_total_size * sizeof(float) << " bytes" << endl;
	d_som = cuda_alloc_float(inputData.som_total_size);
	som = (float *)malloc(inputData.som_total_size * sizeof(float));

	int rotatedImages_size = inputData.numberOfRotations * inputData.neuron_size;
	if (inputData.useFlip) rotatedImages_size *= 2;
	if (inputData.verbose) cout << "  Size of rotated images = " << rotatedImages_size * sizeof(float)<< " bytes" << endl;
	d_rotatedImages = cuda_alloc_float(rotatedImages_size);
	rotatedImages = (float *)malloc(rotatedImages_size * sizeof(float));

	if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << inputData.som_size * sizeof(float) << " bytes" << endl;
	d_euclideanDistanceMatrix = cuda_alloc_float(inputData.som_size);
	euclideanDistanceMatrix = (float *)malloc(inputData.som_size * sizeof(float));

	if (inputData.verbose) cout << "  Size of best rotation matrix = " << inputData.som_size * sizeof(int) << " bytes\n" << endl;
	d_bestRotationMatrix = cuda_alloc_int(inputData.som_size);
	bestRotationMatrix = (int *)malloc(inputData.som_size * sizeof(int));

	if (inputData.verbose) cout << "  Size of image = " << inputData.image_size * sizeof(float) << " bytes\n" << endl;
	d_image = cuda_alloc_float(inputData.image_size);

    // Initialize SOM
	if (inputData.init == ZERO) cuda_fill_zero(d_som, inputData.som_total_size);
	else {
        printf("Random initialization not implemented yet.");
        exit(EXIT_FAILURE);
	}

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

			switch (inputData.algo)
			{
			    case 0:
				{
					generateRotatedImages(rotatedImages, iterImage->getPointerOfFirstPixel(), inputData.numberOfRotations,
						inputData.image_dim, inputData.neuron_dim, inputData.useFlip);

					cuda_copyHostToDevice_float(d_rotatedImages, rotatedImages, inputData.numberOfRotationsAndFlip * inputData.neuron_size);

					cuda_generateEuclideanDistanceMatrix_algo2(d_euclideanDistanceMatrix, d_bestRotationMatrix,
						inputData.som_dim, d_som, inputData.neuron_dim, inputData.numberOfRotations, d_rotatedImages);

					cuda_copyDeviceToHost_float(euclideanDistanceMatrix, d_euclideanDistanceMatrix, inputData.som_size);
					cuda_copyDeviceToHost_int(bestRotationMatrix, d_bestRotationMatrix, inputData.som_size);
					cuda_copyDeviceToHost_float(som, d_som, inputData.som_total_size);

					Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, inputData.som_dim);
					updateNeurons(inputData.som_dim, som, inputData.neuron_dim, rotatedImages, bestMatch, bestRotationMatrix);

					cuda_copyHostToDevice_float(d_som, som, inputData.som_total_size);

					break;
			    }
			    case 1:
				{
					cuda_copyHostToDevice_float(d_image, iterImage->getPointerOfFirstPixel(), inputData.image_size);

					cuda_generateRotatedImages(d_rotatedImages, d_image, inputData.numberOfRotations,
						inputData.image_dim, inputData.neuron_dim, inputData.useFlip);

					cuda_generateEuclideanDistanceMatrix_algo2(d_euclideanDistanceMatrix, d_bestRotationMatrix,
						inputData.som_dim, d_som, inputData.neuron_dim, inputData.numberOfRotations, d_rotatedImages);

					cuda_copyDeviceToHost_float(euclideanDistanceMatrix, d_euclideanDistanceMatrix, inputData.som_size);
					cuda_copyDeviceToHost_int(bestRotationMatrix, d_bestRotationMatrix, inputData.som_size);
					cuda_copyDeviceToHost_float(som, d_som, inputData.som_total_size);

					Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, inputData.som_dim);
					updateNeurons(inputData.som_dim, som, inputData.neuron_dim, rotatedImages, bestMatch, bestRotationMatrix);

					cuda_copyHostToDevice_float(d_som, som, inputData.som_total_size);

					break;
			    }
			    default:
				{
			    	cout << "Unkown algorithm number (" << inputData.algo << ")." << endl;
				    exit(EXIT_FAILURE);
			    }
			}
		}
	}

	// Free memory
    cuda_free(d_image);
    cuda_free(d_bestRotationMatrix);
    cuda_free(d_euclideanDistanceMatrix);
    cuda_free(d_rotatedImages);

	if (inputData.verbose) {
		cout << "  Progress: 100 %\n" << endl;
		cout << "  Write final SOM to " << inputData.resultFilename << " ..." << endl;
	}

	//vector<float> som(inputData.som_total_size);
	//cuda_copyDeviceToHost_float(&som[0], d_som, inputData.som_total_size);
	//writeSOM(&som[0], inputData.som_dim, inputData.neuron_dim, inputData.resultFilename);
	writeSOM(som, inputData.som_dim, inputData.neuron_dim, inputData.resultFilename);

	// Free memory
    cuda_free(d_som);
}
