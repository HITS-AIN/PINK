/**
 * @file   CudaLib/update_neurons.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cuda_runtime.h>

#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/DistributionFunctor.h"
#include "UtilitiesLib/DistanceFunctor.h"

namespace pink {

//! CUDA Kernel Device code updating quadratic self organizing map using gaussian function.
template <unsigned int block_size, class FunctionFunctor, class DistanceFunctor>
__global__ void
update_neurons(float *som, float *rotatedImages, int *bestRotationMatrix, int *bestMatch,
    int neuron_size, FunctionFunctor functionFunctor, DistanceFunctor distanceFunctor,
    float maxUpdateDistance)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= neuron_size) return;

    float distance = distanceFunctor(*bestMatch, blockIdx.y);
    int pos = blockIdx.y * neuron_size + i;

    if (maxUpdateDistance <= 0.0 or distance < maxUpdateDistance)
    {
        som[pos] -= (som[pos] - rotatedImages[bestRotationMatrix[blockIdx.y] * neuron_size + i]) * functionFunctor(distance);
    }
}

} // namespace pink
