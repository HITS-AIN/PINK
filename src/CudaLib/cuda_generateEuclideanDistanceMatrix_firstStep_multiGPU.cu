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
    int GPU_N = cuda_getNumberOfGPUs();
    if (GPU_N > GPU_N_MAX) GPU_N = GPU_N_MAX;

    TGPUplan plan[GPU_N_MAX];

    // Set first device
    cudaSetDevice(0);
    cudaStreamCreate(&plan[0].stream);

    plan[0].size = som_size / GPU_N;
    plan[0].offset = 0;

    // Distribute the remaining neurons
    int rest = som_size % GPU_N;
    if (rest) ++plan[0].size;

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
        plan[i].size = som_size / GPU_N;
        if (rest > i) ++plan[i].size;
        plan[i].offset = plan[i-1].offset + plan[i-1].size;

        // Allocate device memory
        plan[i].d_som = cuda_alloc_float(plan[i].size * neuron_size);
        plan[i].d_rotatedImages = cuda_alloc_float(num_rot * neuron_size);
        plan[i].d_firstStep = cuda_alloc_float(plan[i].size * num_rot);

        // Copy data
        cudaMemcpyPeerAsync(plan[i].d_som, i, plan[0].d_som + plan[i].offset * neuron_size, 0, plan[i].size * neuron_size * sizeof(float), plan[i].stream);
        cudaMemcpyPeerAsync(plan[i].d_rotatedImages, i, plan[0].d_rotatedImages, 0, num_rot * neuron_size * sizeof(float), plan[i].stream);
    }

    // Start kernel
    for (int i = 0; i < GPU_N; ++i)
    {
        // Set device
        cudaSetDevice(i);

        // Setup execution parameters
        dim3 dimBlock(block_size);
        dim3 dimGrid(num_rot, plan[i].size);

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
        if (i != 0)
            cudaMemcpyPeerAsync(plan[0].d_firstStep + plan[i].offset * num_rot, 0, plan[i].d_firstStep, i, plan[i].size * num_rot * sizeof(float), plan[i].stream);

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
            cuda_free(plan[i].d_som);
            cuda_free(plan[i].d_rotatedImages);
            cuda_free(plan[i].d_firstStep);
        }
    }

    cudaSetDevice(0);
    cudaDeviceSynchronize();
}
