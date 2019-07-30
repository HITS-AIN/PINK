/**
 * @file   CudaLib/flip_kernel.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cuda_runtime.h>

namespace pink {

/**
 * CUDA Kernel Device code for flipping an image.
 */
template <typename T>
__global__
void flip_kernel(T *dst, T const *src, uint32_t dim, uint32_t size)
{
    assert(size == dim * dim);

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim or y >= dim) return;

    dst[blockIdx.z*size + (dim-x-1)*dim + y] = src[blockIdx.z*size + x*dim + y];
}

} // namespace pink
