/**
 * @file   CudaLib/updateNeurons_kernel.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/DistributionFunctions.h"

__device__ float d_gaussian(float x, float sigma)
{
    return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * pow((x/sigma),2));
}

__device__ float d_distance_square(int x1, int y1, int x2, int y2)
{
    return sqrt(powf(x1 - x2, 2) + powf(y1 - y2, 2));
}

/**
 * CUDA Kernel Device code updating the self organizing map.
 */
template <unsigned int block_size>
__global__ void
updateNeurons_kernel(float *som, float *rotatedImages, int *bestRotationMatrix, int *bestMatch_x,
    int *bestMatch_y, int neuron_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= neuron_size) return;

    int ij = blockIdx.z*gridDim.y + blockIdx.y;

	float factor = d_gaussian(d_distance_square(*bestMatch_x, *bestMatch_y, blockIdx.z, blockIdx.y), UPDATE_NEURONS_SIGMA) * UPDATE_NEURONS_DAMPING;

	som[ij*neuron_size + i] -= (som[ij*neuron_size + i] - rotatedImages[bestRotationMatrix[ij]*neuron_size + i]) * factor;
}
