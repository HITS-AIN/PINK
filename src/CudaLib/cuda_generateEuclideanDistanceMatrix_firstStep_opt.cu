/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cuda_generateEuclideanDistanceMatrix_firstStep_opt.cu.h"
#include <stdio.h>

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt<512>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt<256>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt<128>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt<64>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt<32>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);
