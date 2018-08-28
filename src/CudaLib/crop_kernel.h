/**
 * @file   CudaLib/crop_kernel.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

namespace pink {

}
/**
 * CUDA Kernel Device code for cropping an image.
 */
template <unsigned int block_size>
__global__ void
crop_kernel(float *dest, float *source, int new_dim, int old_dim)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_dim or y >= new_dim) return;

    int margin = (old_dim - new_dim) * 0.5;
    dest[x*new_dim + y] = source[(x+margin)*old_dim + y+margin];
}

} // namespace pink
