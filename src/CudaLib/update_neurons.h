/**
 * @file   CudaLib/update_neurons.h
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdio>

#include "CudaLib.h"

namespace pink {

/// Find the position where the euclidean distance is minimal between image and neuron.
__global__
void find_best_matching_neuron_kernel(float *euclideanDistanceMatrix, int *bestMatch, int som_size)
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

/// CUDA Kernel Device code updating quadratic self organizing map using gaussian function.
template <unsigned int block_size, class FunctionFunctor, class DistanceFunctor>
__global__
void update_neurons_kernel(float *som, float *rotatedImages, int *bestRotationMatrix, int *bestMatch,
    int neuron_size, FunctionFunctor functionFunctor, DistanceFunctor distanceFunctor,
    float max_update_distance)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= neuron_size) return;

    float distance = distanceFunctor(*bestMatch, blockIdx.y);
    int pos = blockIdx.y * neuron_size + i;

    if (max_update_distance <= 0.0 or distance < max_update_distance)
    {
        som[pos] -= (som[pos] - rotatedImages[bestRotationMatrix[blockIdx.y] * neuron_size + i]) * functionFunctor(distance);
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void update_neurons(thrust::device_vector<T>& d_som, thrust::device_vector<T> const& d_rotated_images,
    thrust::device_vector<uint32_t> const& d_best_rotation_matrix, thrust::device_vector<T> const& d_euclideanDistanceMatrix,
    thrust::device_vector<uint32_t> const& d_bestMatch, int som_width, int som_height, int som_depth, int som_size, int neuron_size,
    DistributionFunction function, float sigma, float damping, float max_update_distance, bool usePBC, int dimensionality)
{
    {
        // Start kernel
        find_best_matching_neuron_kernel<<<1,1>>>(d_euclideanDistanceMatrix, d_bestMatch, som_size);

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel find_best_matching_neuron_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
    {
        // Setup execution parameters
        const uint16_t block_size = 32;
        int gridSize = ceil((float)neuron_size / block_size);
        dim3 dimBlock(block_size);
        dim3 dimGrid(gridSize, som_size);

        // Start kernel
        if (function == DistributionFunction::GAUSSIAN) {
            if (layout == Layout::CARTESIAN) {
                if (usePBC) {
                    if (dimensionality == 1) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), CartesianDistanceFunctor<1, true>(som_width),
                            max_update_distance);
                    } else if (dimensionality == 2) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), CartesianDistanceFunctor<2, true>(som_width, som_height),
                            max_update_distance);
                    } else if (dimensionality == 3) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), CartesianDistanceFunctor<3, true>(som_width, som_height, som_depth),
                            max_update_distance);
                    }
                } else {
                    if (dimensionality == 1) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), CartesianDistanceFunctor<1>(som_width),
                            max_update_distance);
                    } else if (dimensionality == 2) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), CartesianDistanceFunctor<2>(som_width, som_height),
                            max_update_distance);
                    } else if (dimensionality == 3) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), CartesianDistanceFunctor<3>(som_width, som_height, som_depth),
                            max_update_distance);
                    }
                }
            } else if (layout == Layout::HEXAGONAL) {
                update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                    d_bestMatch, neuron_size, GaussianFunctor(sigma, damping), HexagonalDistanceFunctor(som_width),
                    max_update_distance);
            }
        } else if (function == DistributionFunction::MEXICANHAT) {
            if (layout == Layout::CARTESIAN) {
                if (usePBC) {
                    if (dimensionality == 1) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), CartesianDistanceFunctor<1, true>(som_width),
                            max_update_distance);
                    } else if (dimensionality == 2) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), CartesianDistanceFunctor<2, true>(som_width, som_height),
                            max_update_distance);
                    } else if (dimensionality == 3) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), CartesianDistanceFunctor<3, true>(som_width, som_height, som_depth),
                            max_update_distance);
                    }
                } else {
                    if (dimensionality == 1) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), CartesianDistanceFunctor<1>(som_width),
                            max_update_distance);
                    } else if (dimensionality == 2) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), CartesianDistanceFunctor<2>(som_width, som_height),
                            max_update_distance);
                    } else if (dimensionality == 3) {
                        update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                            d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), CartesianDistanceFunctor<3>(som_width, som_height, som_depth),
                            max_update_distance);
                    }
                }
            } else if (layout == Layout::HEXAGONAL) {
                update_neurons_kernel<block_size><<<dimGrid, dimBlock>>>(d_som, d_rotatedImages, d_bestRotationMatrix,
                    d_bestMatch, neuron_size, MexicanHatFunctor(sigma, damping), HexagonalDistanceFunctor(som_width),
                    max_update_distance);
            }
        }

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel update_neurons_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
}

} // namespace pink
