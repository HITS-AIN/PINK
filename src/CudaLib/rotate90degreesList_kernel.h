/**
 * @file   CudaLib/rotateAndCrop_kernel.h
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

namespace pink {

/**
 * CUDA Kernel Device code for special clockwise rotation of 90 degrees of a list of quadratic images.
 */
template <unsigned int block_size>
__global__ void
rotate90degreesList_kernel(float *images, int dim, int size, int offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim or y >= dim) return;

    images[offset + blockIdx.z*size + (dim-y-1)*dim + x] = images[blockIdx.z*size + x*dim + y];
}

} // namespace pink
