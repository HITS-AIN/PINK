/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cuda_generateEuclideanDistanceMatrix_firstStep.cu.h"
#include <cuda_runtime.h>

const int GPU_N_MAX = 4;

struct TGPUplan
{
    float *d_som;
    float *d_rotatedImages;
    float *d_firstStep;

    int size;
    int offset;

    //Stream for asynchronous command execution
    cudaStream_t stream;
};

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix_firstStep_multiGPU(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size, int block_size)
{
    int GPU_N;
    cudaGetDeviceCount(&GPU_N);
    if (GPU_N > GPU_N_MAX) GPU_N = GPU_N_MAX;

    TGPUplan plan[GPU_N_MAX];

    // Set first device
    cudaSetDevice(0);
    cudaStreamCreate(&plan[0].stream);

    // Set size and offset
    if (num_rot % GPU_N)
    {
        fprintf(stderr, "cuda_generateEuclideanDistanceMatrix_firstStep_multiGPU: num_rot not dividable by GPU_N.\n");
        exit(EXIT_FAILURE);
    }

    plan[0].size = num_rot / GPU_N;
    plan[0].offset = 0;

    // Allocate device memory
    plan[0].d_som = d_som;
    plan[0].d_rotatedImages = d_rotatedImages;
    plan[0].d_firstStep = d_firstStep;

    // Create streams for issuing GPU command asynchronously
    for (int i = 1; i < GPU_N; ++i)
    {
        // Set device
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);

        // Set size and offset
        plan[i].size = num_rot / GPU_N;
        plan[i].offset = plan[i-1].offset + plan[i-1].size;

        // Allocate device memory
        plan[i].d_som = cuda_alloc_float(som_size * neuron_size);
        plan[i].d_rotatedImages = cuda_alloc_float(plan[i].size * neuron_size);
        plan[i].d_firstStep = cuda_alloc_float(plan[i].size * som_size);

        // Copy data
        cudaMemcpyPeerAsync(plan[i].d_som, i, plan[0].d_som, 0, som_size * neuron_size, plan[i].stream);
        cudaMemcpyPeerAsync(plan[i].d_rotatedImages, i, plan[0].d_rotatedImages + plan[i].offset, 0, plan[i].size * neuron_size, plan[i].stream);
    }

    // Start kernel
    for (int i = 0; i < GPU_N; ++i)
    {
        // Set device
        cudaSetDevice(i);

        // Setup execution parameters
        dim3 dimBlock(block_size);
        dim3 dimGrid(plan[i].size, som_size);

        switch (block_size)
        {
            case 1024: euclidean_distance_kernel<1024><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            case  512: euclidean_distance_kernel< 512><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            case  256: euclidean_distance_kernel< 256><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            case  128: euclidean_distance_kernel< 128><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            case   64: euclidean_distance_kernel<  64><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            case   32: euclidean_distance_kernel<  32><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            case   16: euclidean_distance_kernel<  16><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotatedImages, plan[i].d_firstStep, neuron_size); break;
            default:
            {
                fprintf(stderr, "cuda_generateEuclideanDistanceMatrix_firstStep: block size (%i) not supported.", block_size);
                exit(EXIT_FAILURE);
            }
        }

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_firstStep (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        // Copy data
        cudaMemcpyPeerAsync(plan[0].d_firstStep, 0, plan[i].d_firstStep, i, plan[i].size, plan[i].stream);

        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to cudaMemcpyPeerAsync (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }

    // Shut down GPU devices
    for (int i = 0; i < GPU_N; ++i)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(plan[i].stream);
        cudaStreamDestroy(plan[i].stream);

        if (i != 0) {
            cudaFree(plan[i].d_som);
            cudaFree(plan[i].d_rotatedImages);
            cudaFree(plan[i].d_firstStep);
        }
    }
}
