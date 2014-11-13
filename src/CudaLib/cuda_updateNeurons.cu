/**
 * @file   CudaLib/cuda_updateNeurons.cu
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cublas_v2.h"
#include <stdio.h>

#define BLOCK_SIZE 32

/**
 * CUDA Kernel Device code
 *
 * Computes multiple rotations of an image. cosine and sin
 */
template <unsigned int block_size>
__global__ void
kernel()
{

//    for (int i = 0; i < som_dim; ++i) {
//        for (int j = 0; j < som_dim; ++j) {
//        	factor = gaussian(distance(bestMatch,Point(i,j)), UPDATE_NEURONS_SIGMA) * UPDATE_NEURONS_DAMPING;
//        	updateSingleNeuron(current_neuron, image + bestRotationMatrix[i*som_dim+j]*image_size, image_size, factor);
//        	current_neuron += image_size;
//    	}
//    }
//}
//
//void updateSingleNeuron(float* neuron, float* image, int image_size, float factor)
//{
//    for (int i = 0; i < image_size; ++i) {
//    	neuron[i] -= (neuron[i] - image[i]) * factor;
//    }

}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void cuda_updateNeurons(float* d_som, float* d_rotatedImages, int* d_bestRotationMatrix, Point bestMatch, int som_dim,
    int neuron_dim, int num_rot)
{
    // Setup execution parameters
	int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(gridSize, gridSize, num_rot);

    // Start kernel
    kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>();

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel cuda_updateNeurons (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
