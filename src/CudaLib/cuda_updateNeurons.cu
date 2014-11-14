/**
 * @file   CudaLib/cuda_updateNeurons.cu
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "updateNeurons_kernel.h"
#include "cublas_v2.h"
#include <float.h>
#include <stdio.h>

#define BLOCK_SIZE 32

/**
 * CUDA Kernel Device code
 *
 * Find the position where the euclidean distance is minimal between image and neuron.
 */
__global__ void
findBestMatchingNeuron_kernel(float *euclideanDistanceMatrix, int *bestMatch_x, int *bestMatch_y, int som_dim)
{
    float minDistance = FLT_MAX;

    for (int i = 0, ij = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j, ++ij) {
			if (euclideanDistanceMatrix[ij] < minDistance) {
				minDistance = euclideanDistanceMatrix[ij];
				*bestMatch_x = i;
				*bestMatch_y = j;
			}
		}
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_updateNeurons(float *d_som, float *d_rotatedImages, int *d_bestRotationMatrix, float *d_euclideanDistanceMatrix,
    int som_dim, int neuron_dim, int num_rot)
{
    int *d_bestMatch_x = cuda_alloc_int(1);
    int *d_bestMatch_y = cuda_alloc_int(1);

    {
    	// Start kernel
        findBestMatchingNeuron_kernel<<<1,1>>>(d_euclideanDistanceMatrix, d_bestMatch_x, d_bestMatch_y, som_dim);

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel findBestMatchingNeuron_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
//    int bestMatch_x;
//    int bestMatch_y;
//    cuda_copyDeviceToHost_int(&bestMatch_x, d_bestMatch_x, 1);
//    cuda_copyDeviceToHost_int(&bestMatch_y, d_bestMatch_y, 1);
//    printf("bestMatch_x = %i\n", bestMatch_x);
//    printf("bestMatch_y = %i\n", bestMatch_y);
    {
		// Setup execution parameters
		int neuron_size = neuron_dim * neuron_dim;
		int gridSize = ceil((float)neuron_size/BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE);
		dim3 dimGrid(gridSize, som_dim, som_dim);

		// Start kernel
		updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
			d_bestMatch_x, d_bestMatch_y, neuron_size);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch CUDA kernel updateNeurons_kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
    }

	cuda_free(d_bestMatch_x);
	cuda_free(d_bestMatch_y);
}
