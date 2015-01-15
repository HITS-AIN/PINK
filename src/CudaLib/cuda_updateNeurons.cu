/**
 * @file   CudaLib/cuda_updateNeurons.cu
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "updateNeurons_kernel.h"
#include <float.h>
#include <stdio.h>

#define BLOCK_SIZE 32

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
void cuda_updateNeurons(float *d_som, float *d_rotatedImages, int *d_bestRotationMatrix, float *d_euclideanDistanceMatrix,
    int* d_bestMatch, int som_width, int som_height, int som_depth, int som_size, int neuron_size, int num_rot,
    Function function, Layout layout, float sigma, float damping, float maxUpdateDistance, bool usePBC, int dimensionality)
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
        if (function == GAUSSIAN) {
            if (layout == QUADRATIC) {
                if (usePBC) {
                    if (dimensionality == 1) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<1, true>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<2, true>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<3, true>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                } else {
                    if (dimensionality == 1) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<1>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<2>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma), CartesianDistanceFunctor<3>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                }
            } else if (layout == HEXAGONAL) {
                updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                    d_bestMatch, neuron_size, GaussianFunctor(sigma), HexagonalDistanceFunctor(som_width),
                    damping, maxUpdateDistance);
            }
        } else if (function == MEXICANHAT) {
            if (layout == QUADRATIC) {
                if (usePBC) {
                    if (dimensionality == 1) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<1, true>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<2, true>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<3, true>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                } else {
                    if (dimensionality == 1) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<1>(som_width),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 2) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<2>(som_width, som_height),
                            damping, maxUpdateDistance);
                    } else if (dimensionality == 3) {
                        updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma), CartesianDistanceFunctor<3>(som_width, som_height, som_depth),
                            damping, maxUpdateDistance);
                    }
                }
            } else if (layout == HEXAGONAL) {
                updateNeurons_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                    d_bestMatch, neuron_size, MexicanHatFunctor(sigma), HexagonalDistanceFunctor(som_width),
                    damping, maxUpdateDistance);
            }
        }

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel updateNeurons_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
}
