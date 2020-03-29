/**
 * @file   CudaLib/rotateAndCrop_kernel.h
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

namespace pink {

/**
 * CUDA Kernel Device code for special clockwise rotation of 90 degrees of a quadratic image.
 */
template <typename T>
__global__ void
rotate90degrees_kernel(thrust::device_ptr<T> dest, thrust::device_ptr<T> source, int dim)
{
    assert(dim > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim or y >= dim) return;

    atomicExch(dest + x*dim + y, source[(dim-y-1)*dim + x]);
}

} // namespace pink
