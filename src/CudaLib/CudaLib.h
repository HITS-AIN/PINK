/**
 * @file   CudaLib.h
 * @brief  Print device properties of GPU cards.
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef CUDALIB_H_
#define CUDALIB_H_

#include "UtilitiesLib/InputData.h"

//! Print CUDA device properties.
void cuda_print_properties();

//! CUDA test routine for image rotation.
void cuda_rotate(int height, int width, float *source, float *dest, float angle);

//! CUDA test routine for euclidean distance calculation.
float cuda_calculateEuclideanDistanceWithoutSquareRoot(float *a, float *b, int length);

//! CUDA test routine for euclidean distance calculation second part.
void cuda_generateEuclideanDistanceMatrix_secondStep(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, float* d_tmp,
    int image_size, int numberOfRotations);

//! Main CUDA host routine for SOM training.
void cuda_trainSelfOrganizingMap(InputData const& inputData);

//! Host routine starting kernel for euclideanDistanceMatrix.
void cuda_generateEuclideanDistanceMatrix(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, int som_dim, float* d_som,
    int image_dim, int num_rot, float* d_image);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
void cuda_generateEuclideanDistanceMatrix_secondStep(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, float* d_tmp,
    int image_size, int num_rot, int red_size);

//! Host routine starting kernel for euclideanDistanceMatrix second algorithm.
void cuda_generateEuclideanDistanceMatrix_algo2(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix, int som_dim, float* d_som,
    int image_dim, int num_rot, float* d_rotatedImages);

//! Host routine starting kernel for euclideanDistanceMatrix second algorithm second part.
void cuda_generateEuclideanDistanceMatrix_algo2_firstStep(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int image_size);

//! Host routine starting kernel for euclideanDistanceMatrix second algorithm second part.
void cuda_generateEuclideanDistanceMatrix_algo2_secondStep(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix,
    float* d_firstStep, int som_size, int num_rot);

//! Basic allocation for device memory.
float* cuda_alloc_float(int size);

//! Basic allocation for device memory.
int* cuda_alloc_int(int size);

//! Fill device memory with zero.
void cuda_fill_zero(float *d, int size);

//! Free device memory.
void cuda_free(float *d);

//! Free device memory.
void cuda_free(int *d);

//! Copy memory from host to device.
void cuda_copyHostToDevice_float(float *dest, float *source, int size);

//! Copy memory from host to device.
void cuda_copyHostToDevice_int(int *dest, int *source, int size);

//! Copy memory from device to host.
void cuda_copyDeviceToHost_float(float *dest, float *source, int size);

//! Copy memory from device to host.
void cuda_copyDeviceToHost_int(int *dest, int *source, int size);

#endif /* CUDALIB_H_ */
