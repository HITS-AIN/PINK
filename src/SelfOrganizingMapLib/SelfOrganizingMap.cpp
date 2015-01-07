/**
 * @file   SelfOrganizingMap.cpp
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include <cmath>
#include <ctype.h>
#include <float.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

void generateRotatedImages(float *rotatedImages, float *image, int num_rot, int image_dim, int neuron_dim,
    bool useFlip, Interpolation interpolation, int numberOfChannels)
{
	int image_size = image_dim * image_dim;
	int neuron_size = neuron_dim * neuron_dim;

	int num_real_rot = num_rot/4;
	float angleStepRadians = num_rot ? 0.5 * M_PI / num_real_rot : 0.0;

	int offset1 = num_real_rot * numberOfChannels * neuron_size;
	int offset2 = 2 * offset1;
	int offset3 = 3 * offset1;

	// Copy original image to first position of image array
    #pragma omp parallel for
	for (int c = 0; c < numberOfChannels; ++c) {
        float *currentImage = image + c*image_size;
        float *currentRotatedImages = rotatedImages + c*neuron_size;
        crop(image_dim, image_dim, neuron_dim, neuron_dim, currentImage, currentRotatedImages);
        rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImages, currentRotatedImages + offset1);
        rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImages + offset1, currentRotatedImages + offset2);
        rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImages + offset2, currentRotatedImages + offset3);
	}

	// Rotate images
    #pragma omp parallel for
	for (int i = 1; i < num_real_rot; ++i) {
	    for (int c = 0; c < numberOfChannels; ++c) {
	        float *currentImage = image + c*image_size;
            float *currentRotatedImage = rotatedImages + (i*numberOfChannels + c)*neuron_size;
            rotateAndCrop(image_dim, image_dim, neuron_dim, neuron_dim, currentImage, currentRotatedImage, i*angleStepRadians, interpolation);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage, currentRotatedImage + offset1);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage + offset1, currentRotatedImage + offset2);
            rotate_90degrees(neuron_dim, neuron_dim, currentRotatedImage + offset2, currentRotatedImage + offset3);
	    }
	}

	// Flip images
	if (useFlip)
	{
		float *flippedRotatedImages = rotatedImages + numberOfChannels * num_rot * neuron_size;

		#pragma omp parallel for
		for (int i = 0; i < num_rot; ++i) {
	        for (int c = 0; c < numberOfChannels; ++c) {
			    flip(neuron_dim, neuron_dim, rotatedImages + (i*numberOfChannels + c)*neuron_size,
			        flippedRotatedImages + (i*numberOfChannels + c)*neuron_size);
	        }
		}
	}
}

void generateEuclideanDistanceMatrix(float *euclideanDistanceMatrix, int *bestRotationMatrix,
    int som_size, float* som, int image_size, int num_rot, float* rotatedImages)
{
	float tmp;
	float* pdist = euclideanDistanceMatrix;
	int* prot = bestRotationMatrix;
    float *psom = NULL;

    for (int i = 0; i < som_size; ++i) euclideanDistanceMatrix[i] = FLT_MAX;

    for (int i = 0; i < som_size; ++i, ++pdist, ++prot) {
        psom = som + i*image_size;
        #pragma omp parallel for private(tmp)
        for (int j = 0; j < num_rot; ++j) {
    	    tmp = calculateEuclideanDistanceWithoutSquareRoot(psom, rotatedImages + j*image_size, image_size);
            #pragma omp critical
    	    if (tmp < *pdist) {
    	    	*pdist = tmp;
                *prot = j;
    	    }
        }
    }
}

int findBestMatchingNeuron(float *euclideanDistanceMatrix, int som_size)
{
    int bestMatch = 0;
    float minDistance = euclideanDistanceMatrix[0];
    for (int i = 1; i < som_size; ++i) {
        if (euclideanDistanceMatrix[i] < minDistance) {
            minDistance = euclideanDistanceMatrix[i];
            bestMatch = i;
        }
    }
    return bestMatch;
}
