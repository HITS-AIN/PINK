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

//! Basic allocation for device memory.
float* cuda_alloc_float(int size);

//! Fill device memory with zero.
void cuda_fill_zero(float *d, int size);

//! Free device memory.
void cuda_free(float *d);

//! Copy memory from host to device.
void cuda_copyHostToDevice_float(float *h, float *d, int size);

//! Copy memory from device to host.
void cuda_copyDeviceToHost_float(float *h, float *d, int size);

#endif /* CUDALIB_H_ */
