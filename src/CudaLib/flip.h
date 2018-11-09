/**
 * @file   CudaLib/flip.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

namespace pink {

/**
 * CUDA Kernel Device code for flipping an image.
 */
template <typename T>
__global__
void flip(thrust::device_ptr<T> dest, thrust::device_ptr<T> source, int dim, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim or y >= dim) return;

    dest[blockIdx.z*size + (dim-x-1)*dim + y] = source[blockIdx.z*size + x*dim + y];
}

} // namespace pink
