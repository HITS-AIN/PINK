/**
 * @file   CudaLib/cuda_generateEuclideanDistanceMatrix_firstStep.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cuda_generateEuclideanDistanceMatrix_firstStep.cu.h"

namespace pink {

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_generateEuclideanDistanceMatrix_firstStep(float *d_som, float *d_rotatedImages,
    float* d_firstStep, int som_size, int num_rot, int neuron_size, int block_size)
{
    // Setup execution parameters
    dim3 dimBlock(block_size);
    dim3 dimGrid(num_rot, som_size);

    // Start kernel
    switch (block_size)
    {
        case 1024: euclidean_distance_kernel<1024><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
        case  512: euclidean_distance_kernel< 512><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
        case  256: euclidean_distance_kernel< 256><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
        case  128: euclidean_distance_kernel< 128><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
        case   64: euclidean_distance_kernel<  64><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
        case   32: euclidean_distance_kernel<  32><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
        case   16: euclidean_distance_kernel<  16><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_firstStep, neuron_size); break;
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
}

} // namespace pink
