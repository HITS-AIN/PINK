/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cuda_generateEuclideanDistanceMatrix_firstStep.cu.h"

template
void cuda_generateEuclideanDistanceMatrix_firstStep<1024>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep<512>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep<256>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep<128>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep<64>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep<32>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);
