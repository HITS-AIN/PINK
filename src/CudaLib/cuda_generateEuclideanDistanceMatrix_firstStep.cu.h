/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep.cu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdio.h>

__device__ void warpReduce(volatile float *data, int tid)
{
    data[tid] += data[tid + 32];
    data[tid] += data[tid + 16];
    data[tid] += data[tid +  8];
    data[tid] += data[tid +  4];
    data[tid] += data[tid +  2];
    data[tid] += data[tid +  1];
}

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void euclidean_distance_kernel(float *som, float *rotatedImages, float *firstStep, int neuron_size)
{
    int tid = threadIdx.x;
    float diff;
    float sum = 0.0f;
    float *psom = som + blockIdx.y * neuron_size;
    float *prot = rotatedImages + blockIdx.x * neuron_size;

    __shared__ float firstStep_local[block_size];

    for (int i = tid; i < neuron_size; i += block_size)
    {
        diff = psom[i] - prot[i];
        sum += diff * diff;
    }

    firstStep_local[tid] = sum;
    __syncthreads();

    // Parallel reduction
    if (block_size >= 512) { if (tid < 256) { firstStep_local[tid] += firstStep_local[tid + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) { firstStep_local[tid] += firstStep_local[tid + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tid <  64) { firstStep_local[tid] += firstStep_local[tid +  64]; } __syncthreads(); }

    //Static loop unrolling for the thread within one warp.
    if (tid < 32) warpReduce(firstStep_local, tid);

    // Copy accumulated local value to global array firstStep
    if (tid == 0) firstStep[blockIdx.x + blockIdx.y * gridDim.x] = firstStep_local[0];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size)
{
    // Setup execution parameters
    dim3 dimBlock(block_size);
    dim3 dimGrid(num_rot, som_size);

    // Start kernel
    euclidean_distance_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_firstStep (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
