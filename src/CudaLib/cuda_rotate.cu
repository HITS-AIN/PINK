/*
 * rotate.cu
 *
 *  Created on: Oct 17, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "cuda_rotate.h"
//#include <stdio.h>
//#include <stdlib.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
rotate_kernel(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_rotate(int height, int width, float *source, float *dest, float angle)
{

}
