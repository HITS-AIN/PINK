/**
 * @file   CudaLib/rotate_90_degrees_list.h
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cuda_runtime.h>

namespace pink {

/**
 * CUDA Kernel Device code for special clockwise rotation of 90 degrees of a list of quadratic images.
 */
template <typename T>
__global__
void rotate_90_degrees_list(T *images, uint32_t dim, uint32_t size, uint32_t offset)
{
    assert(size % dim * dim == 0);

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim or y >= dim) return;

    images[offset + blockIdx.z*size + x*dim + y] = images[blockIdx.z*size + (dim-y-1)*dim + x];
}

} // namespace pink
