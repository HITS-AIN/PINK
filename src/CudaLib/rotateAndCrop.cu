/**
 * @file   rotateAndCrop.cu
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include <stdio.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void rotateAndCrop_kernel(float * Destination, const float sintheta,
    const float costheta, const int width)
{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	float tx = float(x)*costheta-float(y)*sintheta;
//	float ty = float(x)*sintheta+float(y)*costheta;
//
//	if(x<width && y<width) {
//		Destination[x*width+y]=tex2D(Source_texture, tx+0.5f,ty+0.5f);
//	}
}
