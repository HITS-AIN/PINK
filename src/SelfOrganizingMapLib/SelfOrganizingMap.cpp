/**
 * @file   SelfOrganizingMap.cpp
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "UtilitiesLib/DistributionFunctions.h"
#include <ctype.h>
#include <float.h>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

std::ostream& operator << (std::ostream& os, Layout layout)
{
	if (layout == QUADRATIC) os << "quadratic";
	else if (layout == HEXAGONAL) os << "hexagonal";
	else os << "undefined";
	return os;
}

std::ostream& operator << (std::ostream& os, SOMInitialization init)
{
	if (init == ZERO) os << "zero";
	else if (init == RANDOM) os << "random";
	else os << "undefined";
	return os;
}

void generateRotatedImages(float *rotatedImages, float *image, int numberOfRotations, int image_dim, int neuron_dim, bool useFlip)
{
	int image_size = image_dim * image_dim;
	int neuron_size = neuron_dim * neuron_dim;
	float angleStepRadians = numberOfRotations ? 2.0 * M_PI / numberOfRotations : 0.0;

	// Copy original image on first position
	crop(image_dim, image_dim, neuron_dim, neuron_dim, image, rotatedImages);

	// Rotate unflipped image
    #pragma omp parallel for
	for (int i = 1; i < numberOfRotations; ++i) {
		rotateAndCrop(image_dim, image_dim, neuron_dim, neuron_dim, image, rotatedImages + i*neuron_size, i*angleStepRadians);
	}

	if (useFlip)
	{
		// Flip image
		float *flippedImage = (float *)malloc(image_size * sizeof(float));
		flip(image_dim, image_dim, image, flippedImage);

		float *flippedRotatedImage = rotatedImages + numberOfRotations * neuron_size;
		crop(image_dim, image_dim, neuron_dim, neuron_dim, flippedImage, flippedRotatedImage);

		// Rotate flipped image
		#pragma omp parallel for
		for (int i = 1; i < numberOfRotations; ++i)	{
			rotateAndCrop(image_dim, image_dim, neuron_dim, neuron_dim, flippedImage, flippedRotatedImage + i*neuron_size, i*angleStepRadians);
		}

		free(flippedImage);
	}
}

void generateEuclideanDistanceMatrix(float *euclideanDistanceMatrix, int *bestRotationMatrix, int som_dim, float* som,
	int image_dim, int num_rot, float* rotatedImages)
{
	int som_size = som_dim * som_dim;
	int image_size = image_dim * image_dim;

	float tmp;
	float* pdist = euclideanDistanceMatrix;
	int* prot = bestRotationMatrix;

    for (int i = 0; i < som_size; ++i) euclideanDistanceMatrix[i] = FLT_MAX;

    for (int i = 0; i < som_size; ++i, ++pdist, ++prot) {
        #pragma omp parallel for private(tmp)
        for (int j = 0; j < num_rot; ++j) {
    	    tmp = calculateEuclideanDistanceWithoutSquareRoot(som + i*image_size, rotatedImages + j*image_size, image_size);
            #pragma omp critical
    	    if (tmp < *pdist) {
    	    	*pdist = tmp;
                *prot = j;
    	    }
        }
    }
}

Point findBestMatchingNeuron(float *euclideanDistanceMatrix, int som_dim)
{
	int som_size = som_dim * som_dim;
    float minDistance = FLT_MAX;
    Point bestMatch;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
			if (euclideanDistanceMatrix[i*som_dim+j] < minDistance) {
				minDistance = euclideanDistanceMatrix[i*som_dim+j];
				bestMatch.x = i;
				bestMatch.y = j;
			}
		}
    }

    return bestMatch;
}

void updateNeurons(int som_dim, float* som, int image_dim, float* image, Point const& bestMatch, int *bestRotationMatrix)
{
	float factor;
	int image_size = image_dim * image_dim;
	float *current_neuron = som;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
        	factor = gaussian(distance(bestMatch,Point(i,j)), UPDATE_NEURONS_SIGMA) * UPDATE_NEURONS_DAMPING;
        	updateSingleNeuron(current_neuron, image + bestRotationMatrix[i*som_dim+j]*image_size, image_size, factor);
        	current_neuron += image_size;
    	}
    }
}

void updateSingleNeuron(float* neuron, float* image, int image_size, float factor)
{
    for (int i = 0; i < image_size; ++i) {
    	neuron[i] -= (neuron[i] - image[i]) * factor;
    }
}
