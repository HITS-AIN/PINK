/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep.cu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdio.h>

namespace pink {

/**
 * CUDA Kernel Device code
 *
 * Static loop unrolling for the thread within one warp.
 */
template <unsigned int block_size>
__device__ void warpReduce(volatile float *data, int tid)
{
    if (block_size >= 64) data[tid] += data[tid + 32];
    if (block_size >= 32) data[tid] += data[tid + 16];
    if (block_size >= 16) data[tid] += data[tid +  8];
    if (block_size >=  8) data[tid] += data[tid +  4];
    if (block_size >=  4) data[tid] += data[tid +  2];
    if (block_size >=  2) data[tid] += data[tid +  1];
}

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void euclidean_distance_kernel(float *som, float *rotatedImages, float *firstStep, int neuron_size)
{
    int tid = threadIdx.x;
    float diff;
    float sum = 0.0f;
    float *psom = som + blockIdx.y * neuron_size;
    float *prot = rotatedImages + blockIdx.x * neuron_size;

    __shared__ float firstStep_local[block_size];

    for (int i = tid; i < neuron_size; i += block_size)
    {
        diff = psom[i] - prot[i];
        sum += diff * diff;
    }

    firstStep_local[tid] = sum;
    __syncthreads();

    // Parallel reduction
    if (block_size >= 512) { if (tid < 256) { firstStep_local[tid] += firstStep_local[tid + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) { firstStep_local[tid] += firstStep_local[tid + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tid <  64) { firstStep_local[tid] += firstStep_local[tid +  64]; } __syncthreads(); }

    // Static loop unrolling for the thread within one warp.
    if (tid < 32) warpReduce<block_size>(firstStep_local, tid);

    // Copy accumulated local value to global array firstStep
    if (tid == 0) firstStep[blockIdx.x + blockIdx.y * gridDim.x] = firstStep_local[0];
}

} // namespace pink
