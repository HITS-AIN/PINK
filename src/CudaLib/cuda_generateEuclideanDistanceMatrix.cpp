/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix.cpp
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include <stdio.h>

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix(float *d_euclideanDistanceMatrix, int *d_bestRotationMatrix,
    int som_dim, float* d_som, int image_dim, int num_rot, float* d_rotatedImages, int numberOfChannels)
{
	unsigned int image_size = image_dim * image_dim;
	unsigned int som_size = som_dim * som_dim;

    float *d_firstStep = cuda_alloc_float(som_size * num_rot);

    // First step ...
    cuda_generateEuclideanDistanceMatrix_firstStep(d_som, d_rotatedImages,
        d_firstStep, som_size, num_rot, numberOfChannels * image_size, 128);

//    float firstStep;
//    cuda_copyDeviceToHost_float(&firstStep, d_firstStep, 1);
//    printf("firstStep = %f", firstStep);

    // Second step ...
    cuda_generateEuclideanDistanceMatrix_secondStep(d_euclideanDistanceMatrix,
        d_bestRotationMatrix, d_firstStep, som_size, num_rot);

    cuda_free(d_firstStep);
}
