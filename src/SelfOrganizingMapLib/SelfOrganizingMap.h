/**
 * @file   SelfOrganizingMap.h
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef SELFORGANIZINGMAP_H_
#define SELFORGANIZINGMAP_H_

#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/Point.h"
#include <iostream>

#define UPDATE_NEURONS_SIGMA     1.1
#define UPDATE_NEURONS_DAMPING   0.2

//! Main CPU based routine for SOM training.
void trainSelfOrganizingMap(InputData const& inputData);

void generateRotatedImages(float *rotatedImages, float *image, int numberOfRotations, int image_dim, int neuron_dim);

void generateEuclideanDistanceMatrix(float *euclideanDistanceMatrix, int *bestRotationMatrix, int som_dim, float* som,
	int image_dim, int numberOfRotations, float* image);

/**
 * Returns the position of the best matching neuron.
 */
Point findBestMatchingNeuron(float *similarityMatrix, int som_dim);

/**
 * Updating SOM
 */
void updateNeurons(int som_dim, float* som, int image_dim, float* image, Point const& bestMatch, int *bestRotationMatrix);

void updateSingleNeuron(float* neuron, float* image, int image_size, float factor);

float distance(Point pos1, Point pos2);

float mexicanHat(float x, float sigma);

float gaussian(float x, float sigma);

#endif /* SELFORGANIZINGMAP_H_ */
