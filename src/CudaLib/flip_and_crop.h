/**
 * @file   CudaLib/flip_and_crop.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cuda_runtime.h>

namespace pink {

/**
 * CUDA Kernel Device code for combined flipping and cropping an image.
 */
template <unsigned int block_size>
__global__ void
flip_and_crop(float *dest, float *source, int new_dim, int old_dim)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_dim or y >= new_dim) return;

    int margin = (old_dim - new_dim) * 0.5;

    dest[(new_dim-x-1)*new_dim + y] = source[(x+margin)*old_dim + y+margin];
}

} // namespace pink
