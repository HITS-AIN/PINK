/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep_opt2.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cuda_generateEuclideanDistanceMatrix_firstStep_opt2.cu.h"

namespace pink {

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2<512>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2<256>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2<128>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2<64>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

template
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2<32>(float *d_som, float *d_rotatedImages,
   float* d_firstStep, int som_size, int num_rot, int image_size);

} // namespace pink
