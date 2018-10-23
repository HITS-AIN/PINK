/**
 * @file   CudaLib/crop.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

namespace pink {

/**
 * CUDA Kernel Device code for cropping an image.
 */
template <typename T>
__global__ void
crop(thrust::device_ptr<T> dest, thrust::device_ptr<T> source, int new_dim, int old_dim)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_dim or y >= new_dim) return;

    int margin = (old_dim - new_dim) * 0.5;
    dest[x*new_dim + y] = source[(x+margin)*old_dim + y+margin];
}

} // namespace pink
