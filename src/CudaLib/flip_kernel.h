/**
 * @file   CudaLib/flip_kernel.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

/**
 * CUDA Kernel Device code for flipping an image.
 */
template <unsigned int block_size>
__global__ void
flip_kernel(float *dest, float *source, int dim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dim or y >= dim) return;

    dest[(dim-x-1)*dim + y] = source[x*dim + y];
}
