/**
 * @file   CudaLib/crop.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cuda_runtime.h>

namespace pink {

/**
 * CUDA Kernel Device code for cropping an image.
 */
template <typename T>
__global__
void crop(T *dst, T const *src, uint32_t new_dim, uint32_t old_dim)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_dim or y >= new_dim) return;

    uint32_t margin = (old_dim - new_dim) * 0.5;
    dst[x*new_dim + y] = src[(x+margin)*old_dim + y+margin];
}

} // namespace pink
