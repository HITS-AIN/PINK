/**
 * @file   CudaLib.h
 * @brief  Print device properties of GPU cards.
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef CUDALIB_H_
#define CUDALIB_H_

#include "UtilitiesLib/InputData.h"

void cuda_print_properties();

void cuda_rotate(int height, int width, float *source, float *dest, float angle);

float cuda_calculateEuclideanDistanceWithoutSquareRoot(float *a, float *b, int length);

//! Main CUDA host routine for SOM training.
void cuda_trainSelfOrganizingMap(InputData const& inputData);

#endif /* CUDALIB_H_ */
