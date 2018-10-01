/**
 * @file   CudaLib/CudaLib.h
 * @brief  Header for all CUDA functions
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/Point.h"

namespace pink {

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

//! Main CUDA host routine for SOM mapping.
void cuda_mapping(InputData const& inputData);

//! Host routine starting kernel for euclideanDistanceMatrix.
void cuda_generateEuclideanDistanceMatrix(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix,
    int som_size, float* d_som, int image_size, int num_rot, float* d_rotatedImages, int block_size_1,
    bool useMultipleGPUs);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
void cuda_generateEuclideanDistanceMatrix_firstStep(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int image_size, int block_size);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
void cuda_generateEuclideanDistanceMatrix_firstStep_multiGPU(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int image_size, int block_size);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int image_size);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int image_size);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt3(float *d_som, float *d_rotatedImages,
    int som_size, int num_rot, int image_size);

//! Host routine starting kernel for euclideanDistanceMatrix second part.
void cuda_generateEuclideanDistanceMatrix_secondStep(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix,
    float* d_firstStep, int som_size, int num_rot);

//! Host routine starting kernel for rotated images.
void generate_rotated_images_gpu(float* d_rotatedImages, float* d_image, int num_rot, int image_dim, int neuron_dim,
    bool flip, Interpolation interpolation, float *d_cosAlpha, float *d_sinAlpha, int numberOfChannels);

//! Host routine starting kernel for updating neurons.
void update_neurons(float *d_som, float *d_rotatedImages, int *d_bestRotationMatrix, float *d_euclideanDistanceMatrix,
    int* d_bestMatch, int som_width, int som_height, int som_depth, int som_size, int neuron_size,
    DistributionFunction function, Layout layout, float sigma, float damping, float maxUpdateDistance, bool usePBC, int dimensionality);

//! Prepare trigonometric values
void trigonometricValues(float **d_cosAlpha, float **d_sinAlpha, int num_rot);

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

//! Return number of GPUs.
int cuda_getNumberOfGPUs();

//! Set GPU device number.
void cuda_setDevice(int number);

} // namespace pink
