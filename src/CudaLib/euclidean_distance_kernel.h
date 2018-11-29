/**
 * @file   CudaLib/euclidean_distance_kernel.h
 * @date   Nov 23, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cstdint>

namespace pink {

/// CUDA device kernel for reducing a data array with the length of 64 by
/// static loop unrolling for the thread within one warp
template <typename DataType>
__device__
void warp_reduce_64(volatile DataType *data, int tid)
{
    data[tid] += data[tid + 32];
    data[tid] += data[tid + 16];
    data[tid] += data[tid +  8];
    data[tid] += data[tid +  4];
    data[tid] += data[tid +  2];
    data[tid] += data[tid +  1];
}

/// CUDA device kernel to computes the euclidean distance of two arrays
/// using a reduced type to calculate the euclidean distance
template <uint16_t block_size, typename DataType, typename EuclideanType>
__global__
void euclidean_distance_kernel(EuclideanType const *som, EuclideanType const *rotated_images,
    DataType *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<512>(float const *som, float const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<256>(float const *som, float const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<128>(float const *som, float const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<64>(float const *som, float const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<512>(uint8_t const *som, uint8_t const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<256>(uint8_t const *som, uint8_t const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<128>(uint8_t const *som, uint8_t const *rotated_images, float *first_step, uint32_t neuron_size);

template <>
__global__
void euclidean_distance_kernel<64>(uint8_t const *som, uint8_t const *rotated_images, float *first_step, uint32_t neuron_size);

} // namespace pink
