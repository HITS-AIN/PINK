/**
 * @file   cuda_rotate.cu
 * @date   Oct 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "cuda_rotate.h"
#include <stdio.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
rotate_kernel(const float *source, float *dest, int height, int width, float alpha)
{
    int i  = blockDim.x * blockIdx.x + threadIdx.x;
    int j  = blockDim.y * blockIdx.y + threadIdx.y;
    int is = blockDim.x * gridDim.x;
    int js = blockDim.y * gridDim.y;

    int x0, x1, x2, y0, y1, y2;
    const float cosAlpha = cos(alpha);
    const float sinAlpha = sin(alpha);

    x0 = width / 2;
    y0 = height / 2;

    for (x1 = i; x1 < width; x1 += is) {
        for (y1 = j; y1 < height; y1 += js) {
        	x2 = (x1 - x0) * cosAlpha - (y1 - y0) * sinAlpha + x0;
        	y2 = (x1 - x0) * sinAlpha + (y1 - y0) * cosAlpha + y0;
            if (x2 > -1 && x2 < width && y2 > -1 && y2 < height) dest[x2*height + y2] = source[x1*height + y1];
        }
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_rotate(int height, int width, float *source, float *dest, float alpha)
{
    unsigned int size = height * width;
    unsigned int sizeInBytes = size * sizeof(float);

    // Allocate device memory
    float *d_source, *d_dest;

    cudaError_t error;

    error = cudaMalloc((void **) &d_source, sizeInBytes);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_source returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_dest, sizeInBytes);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_dest returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_source, source, sizeInBytes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_source, source) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Setup execution parameters
    const unsigned int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(height/blockSize, width/blockSize);
    //dim3 dimBlock(1,1);
    //dim3 dimGrid(1,1);

    printf("Starting CUDA Kernel with (%i,%i,%i) blocks and (%i,%i,%i) threads ...\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);

    rotate_kernel<<<dimGrid, dimBlock>>>(d_source, d_dest, height, width, alpha);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Copy the device result vector in device memory to the host result vector in host memory.
    error = cudaMemcpy(dest, d_dest, sizeInBytes, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy d_dest to host (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    error = cudaFree(d_dest);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_dest (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_source);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_source (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
