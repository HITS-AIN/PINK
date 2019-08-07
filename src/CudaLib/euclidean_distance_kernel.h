/**
 * @file   CudaLib/euclidean_distance_kernel.h
 * @date   Nov 23, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <sm_61_intrinsics.h>

#include "UtilitiesLib/ipow.h"

namespace pink {

/// CUDA device kernel for reducing a data array with the length of 64 by
/// static loop unrolling for the thread within one warp
template <typename T>
__device__
void warp_reduce_64(volatile T *data, int tid)
{
    assert(tid >= 0);
    assert(tid < 32);

    data[tid] += data[tid + 32];
    data[tid] += data[tid + 16];
    data[tid] += data[tid +  8];
    data[tid] += data[tid +  4];
    data[tid] += data[tid +  2];
    data[tid] += data[tid +  1];
}

template <typename T>
struct scale;

template <>
struct scale<float>
{
    __device__ void operator () (float& sum) const { /* nothing to do */ }
};

template <>
struct scale<uint16_t>
{
    static constexpr uint32_t range = ipow(2, std::numeric_limits<uint16_t>::digits) - 1;
    static constexpr uint32_t factor = range * range;
    __device__ void operator () (float& sum) const { sum /= factor; }
};

template <>
struct scale<uint8_t>
{
    static constexpr uint32_t range = ipow(2, std::numeric_limits<uint8_t>::digits) - 1;
    static constexpr uint32_t factor = range * range;
    __device__ void operator () (float& sum) const { sum /= factor; }
};

/// CUDA device kernel to computes the euclidean distance of two arrays
/// using a reduced type to calculate the euclidean distance
template <uint32_t block_size, typename DataType, typename EuclideanType>
__global__ static
void euclidean_distance_kernel(EuclideanType const *som, EuclideanType const *rotated_images,
    DataType *first_step, uint32_t neuron_size)
{
    int tid = threadIdx.x;
    DataType sum = 0.0;
    EuclideanType const *psom = som + blockIdx.y * neuron_size;
    EuclideanType const *prot = rotated_images + blockIdx.x * neuron_size;

    __shared__ DataType first_step_local[block_size];

    for (uint32_t i = tid; i < neuron_size; i += block_size)
    {
        DataType diff = psom[i] - prot[i];
        sum += diff * diff;
    }

    scale<EuclideanType>()(sum);
    first_step_local[tid] = sum;
    __syncthreads();

    // Parallel reduction
    if (block_size > 256 and tid < 256) {
        first_step_local[tid] += first_step_local[tid + 256];
        __syncthreads();
    }
    if (block_size > 128 and tid < 128) {
        first_step_local[tid] += first_step_local[tid + 128];
        __syncthreads();
    }
    if (block_size > 64 and tid < 64) {
        first_step_local[tid] += first_step_local[tid + 64];
        __syncthreads();
    }

    // Static loop unrolling for the thread within one warp.
    if (tid < 32) warp_reduce_64(first_step_local, tid);

    // Copy accumulated local value to global array first_step
    if (tid == 0) first_step[blockIdx.x + blockIdx.y * gridDim.x] = first_step_local[0];
}

} // namespace pink
