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

void generateEuclideanDistanceMatrix(float *euclideanDistanceMatrix, int *bestRotationMatrix, int som_dim, float* som,
	int image_dim, int num_rot, float* rotatedImages, int numberOfChannels)
{
	int som_size = som_dim * som_dim;
	int image_size = image_dim * image_dim;

	float tmp;
	float* pdist = euclideanDistanceMatrix;
	int* prot = bestRotationMatrix;

    for (int i = 0; i < som_size; ++i) euclideanDistanceMatrix[i] = FLT_MAX;

    int channel_image_size = numberOfChannels * image_size;
    float *psom = NULL;

    for (int i = 0; i < som_size; ++i, ++pdist, ++prot) {
        psom = som + i*channel_image_size;
        #pragma omp parallel for private(tmp)
        for (int j = 0; j < num_rot; ++j) {
    	    tmp = calculateEuclideanDistanceWithoutSquareRoot(psom, rotatedImages + j*channel_image_size, channel_image_size);
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
    float minDistance = euclideanDistanceMatrix[0];
    Point bestMatch(0,0);
    for (int ij = 1; ij < som_size; ++ij) {
        if (euclideanDistanceMatrix[ij] < minDistance) {
            minDistance = euclideanDistanceMatrix[ij];
            bestMatch.x = ij / som_dim;
            bestMatch.y = ij % som_dim;
        }
    }

    return bestMatch;
}

void updateNeurons(int som_dim, float* som, int image_dim, float* image, Point const& bestMatch,
    int *bestRotationMatrix, int numberOfChannels, std::shared_ptr<DistributionFunctorBase> const& ptrDistributionFunctor,
    std::shared_ptr<DistanceFunctorBase> const& ptrDistanceFunctor, float damping, float maxUpdateDistance)
{
    float distance, factor;
    int image_size = image_dim * image_dim;
    float *current_neuron = som;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
            distance = (*ptrDistanceFunctor)(bestMatch.x, bestMatch.y,i,j);
            if (maxUpdateDistance <= 0.0 or distance < maxUpdateDistance) {
                factor = (*ptrDistributionFunctor)(distance) * damping;
                updateSingleNeuron(current_neuron, image + bestRotationMatrix[i*som_dim+j] * numberOfChannels * image_size, numberOfChannels * image_size, factor);
                current_neuron += numberOfChannels * image_size;
            }
        }
    }
}

void updateSingleNeuron(float* neuron, float* image, int image_size, float factor)
{
    for (int i = 0; i < image_size; ++i) {
    	neuron[i] -= (neuron[i] - image[i]) * factor;
    }
}
