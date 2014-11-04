/**
 * @file   cuda_functions.cu
 * @brief  Basic functions of CUDA.
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include <stdio.h>

float* cuda_alloc_float(int size)
{
	float *d;

	cudaError_t error = cudaMalloc((void **) &d, size * sizeof(float));

    if (error != cudaSuccess)
    {
		fprintf(stderr, "cudaMalloc failed (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return d;
}
void cuda_fill_zero(float* d, int size)
{
	cudaError_t error = cudaMemset(d, 0, size * sizeof(float));

    if (error != cudaSuccess)
    {
		fprintf(stderr, "cudaMemset failed (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void cuda_free(float* d)
{
	cudaError_t error = cudaFree(d);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void cuda_copyHostToDevice_float(float *h, float *d, int size)
{
	cudaError_t error = cudaMemcpy(d, h, size * sizeof(float), cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
		fprintf(stderr, "cudaMemcpy failed (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void cuda_copyDeviceToHost_float(float *d, float *h, int size)
{
	cudaError_t error = cudaMemcpy(h, d, size * sizeof(float), cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}
