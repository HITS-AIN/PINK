/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep_opt2.cu.h
 * @date   Dec 8, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdio.h>

namespace pink {

/**
 * CUDA Kernel Device code
 *
 * Computes the euclidean distance of two arrays.
 */
template <unsigned int block_size>
__global__ void euclidean_distance_opt2_kernel(float *som, float *rotatedImages, float *firstStep, int neuron_size)
{
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float *psom = som + blockIdx.z * neuron_size + blockIdx.x * block_size;
    float *prot = rotatedImages + blockIdx.y * neuron_size + blockIdx.x * block_size;

    __shared__ float firstStep_local[block_size];

    if (i < neuron_size) {
        float diff = psom[tid] - prot[tid];
        firstStep_local[tid] = diff * diff;
    } else {
        firstStep_local[tid] = 0.0f;
    }

    __syncthreads();

    // Parallel reduction
    if (block_size >= 512) { if (tid < 256) { firstStep_local[tid] += firstStep_local[tid + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) { firstStep_local[tid] += firstStep_local[tid + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tid <  64) { firstStep_local[tid] += firstStep_local[tid +  64]; } __syncthreads(); }
    if (block_size >=  64) { if (tid <  32) { firstStep_local[tid] += firstStep_local[tid +  32]; } __syncthreads(); }
    if (block_size >=  32) { if (tid <  16) { firstStep_local[tid] += firstStep_local[tid +  16]; } __syncthreads(); }
    if (block_size >=  16) { if (tid <   8) { firstStep_local[tid] += firstStep_local[tid +   8]; } __syncthreads(); }
    if (block_size >=   8) { if (tid <   4) { firstStep_local[tid] += firstStep_local[tid +   4]; } __syncthreads(); }
    if (block_size >=   4) { if (tid <   2) { firstStep_local[tid] += firstStep_local[tid +   2]; } __syncthreads(); }
    if (block_size >=   2) { if (tid <   1) { firstStep_local[tid] += firstStep_local[tid +   1]; } __syncthreads(); }

    // Copy accumulated local value to global array firstStep
    if (tid == 0) atomicAdd(firstStep + blockIdx.y + blockIdx.z * gridDim.y, firstStep_local[0]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <unsigned int block_size>
void cuda_generateEuclideanDistanceMatrix_firstStep_opt2(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size)
{
    // Setup execution parameters
    int grid_size = ceil((float)neuron_size/block_size);
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size, num_rot, som_size);

    // Start kernel
    euclidean_distance_opt2_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_firstStep_opt2 (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

} // namespace pink
