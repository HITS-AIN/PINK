/**
 * @file   CudaLib/update_neurons.cu
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <float.h>
#include <stdio.h>

#include "CudaLib.h"
#include "update_neurons.h"

#define BLOCK_SIZE 32

namespace pink {

/**
 * CUDA Kernel Device code
 *
 * Find the position where the euclidean distance is minimal between image and neuron.
 */
__global__ void
findBestMatchingNeuron_kernel(float *euclideanDistanceMatrix, int *bestMatch, int som_size)
{
    *bestMatch = 0;
    float minDistance = euclideanDistanceMatrix[0];
    for (int i = 1; i < som_size; ++i) {
        if (euclideanDistanceMatrix[i] < minDistance) {
            minDistance = euclideanDistanceMatrix[i];
            *bestMatch = i;
        }
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
void update_neurons(float *d_som, float *d_rotatedImages, int *d_bestRotationMatrix, float *d_euclideanDistanceMatrix,
    int* d_bestMatch, int som_width, int som_height, int som_depth, int som_size, int neuron_size,
	DistributionFunction function, Layout layout, float sigma, float damping, float maxUpdateDistance, bool usePBC, int dimensionality)
{
    {
        // Start kernel
        findBestMatchingNeuron_kernel<<<1,1>>>(d_euclideanDistanceMatrix, d_bestMatch, som_size);

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel findBestMatchingNeuron_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
    {
        // Setup execution parameters
        int gridSize = ceil((float)neuron_size/BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(gridSize, som_size);

        // Start kernel
        if (function == DistributionFunction::GAUSSIAN) {
            if (layout == Layout::CARTESIAN) {
                if (usePBC) {
                    if (dimensionality == 1) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<1, true>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<2, true>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<3, true>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                } else {
                    if (dimensionality == 1) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<1>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<2>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<3>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                }
            } else if (layout == Layout::HEXAGONAL) {
                update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                    d_bestMatch, neuron_size, GaussianFunctor(sigma), HexagonalDistanceFunctor(som_width),
                    damping, maxUpdateDistance);
            }
        } else if (function == DistributionFunction::MEXICANHAT) {
            if (layout == Layout::CARTESIAN) {
                if (usePBC) {
                    if (dimensionality == 1) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<1, true>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<2, true>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<3, true>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                } else {
                    if (dimensionality == 1) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<1>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<2>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<3>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                }
            } else if (layout == Layout::HEXAGONAL) {
                update_neurons<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                    d_bestMatch, neuron_size, MexicanHatFunctor(sigma), HexagonalDistanceFunctor(som_width),
                    damping, maxUpdateDistance);
            }
        }

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel update_neurons (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
}

} // namespace pink
