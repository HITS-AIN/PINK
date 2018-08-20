/**
 * @file   SelfOrganizingMapLib/SelfOrganizingMap.h
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include "ImageProcessingLib/ImageProcessing.h"
#include "UtilitiesLib/DistanceFunctor.h"
#include "UtilitiesLib/DistributionFunctor.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/Point.h"
#include <iostream>
#include <memory>

void generateRotatedImages(float *rotatedImages, float *image, int numberOfRotations, int image_dim, int neuron_dim,
    bool useFlip, Interpolation interpolation, int numberOfChannels);

void generateEuclideanDistanceMatrix(float *euclideanDistanceMatrix, int *bestRotationMatrix, int som_size, float* som,
    int image_size, int numberOfRotations, float* image);

//! Returns the position of the best matching neuron (lowest euclidean distance).
int findBestMatchingNeuron(float *euclideanDistanceMatrix, int som_size);
