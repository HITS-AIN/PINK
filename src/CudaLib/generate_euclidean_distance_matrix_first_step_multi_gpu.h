/**
 * @file   CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <omp.h>
#include <thread>
#include <thrust/device_vector.h>

#include "euclidean_distance_kernel.h"

namespace pink {

template <typename DataType, typename EuclideanType>
struct Plan
{
    Plan(thrust::device_vector<EuclideanType> const& d_som,
         thrust::device_vector<EuclideanType> const& d_rotated_images,
         thrust::device_vector<DataType> d_first_step)
     : d_som(d_som),
       d_rotated_images(d_rotated_images),
       d_first_step(d_first_step)
    {
        cudaStreamCreate(&stream);
    }

    ~Plan()
    {
        cudaStreamDestroy(stream);
    }

    thrust::device_vector<EuclideanType> d_som;
    thrust::device_vector<EuclideanType> d_rotated_images;
    thrust::device_vector<DataType> d_first_step;

    uint32_t size;
    uint32_t offset;

    // CUDA stream for asynchronous command execution
    cudaStream_t stream;
};

/// Host function that prepares data array and passes it to the CUDA kernel.
template <typename DataType, typename EuclideanType>
void generate_euclidean_distance_matrix_first_step_multi_gpu(thrust::device_vector<EuclideanType> const& d_som,
    thrust::device_vector<EuclideanType> const& d_rotated_images, thrust::device_vector<DataType> d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size, uint32_t neuron_size, uint16_t block_size)
{
    auto&& gpu_ids = cuda_get_gpu_ids();
    auto&& number_of_gpus = cuda_get_gpu_ids().size();
    auto&& number_of_threads = omp_get_num_threads();

    if (number_of_threads < number_of_gpus)
        throw pink::exception("Number of CPU threads must not be smaller than the number of GPU devices");

    std::vector<std::thread> workers;

#if 0
    std::vector<Plan<DataType, EuclideanType>> plan;

    // Set first device
    cudaSetDevice(gpu_ids[0]);
    cudaStreamCreate(&plan[0].stream);

    plan[0].size = som_size / number_of_gpus;
    plan[0].offset = 0;

    // Distribute the remaining neurons
    int rest = som_size % number_of_gpus;
    if (rest) ++plan[0].size;

    // Allocate device memory
    plan[0].d_som = d_som;
    plan[0].d_rotated_images = d_rotated_images;
    plan[0].d_first_step = d_first_step;

    // Create streams for issuing GPU command asynchronously
    for (int i = 1; i < number_of_gpus; ++i)
    {
        // Set device
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);

        // Set size and offset
        plan[i].size = som_size / number_of_gpus;
        if (rest > i) ++plan[i].size;
        plan[i].offset = plan[i-1].offset + plan[i-1].size;

        // Allocate device memory
        plan[i].d_som = cuda_alloc_float(plan[i].size * neuron_size);
        plan[i].d_rotated_images = cuda_alloc_float(number_of_spatial_transformations * neuron_size);
        plan[i].d_first_step = cuda_alloc_float(plan[i].size * number_of_spatial_transformations);

        // Copy data
        cudaMemcpyPeerAsync(plan[i].d_som, i, plan[0].d_som + plan[i].offset * neuron_size, 0, plan[i].size * neuron_size * sizeof(float), plan[i].stream);
        cudaMemcpyPeerAsync(plan[i].d_rotated_images, i, plan[0].d_rotated_images, 0, number_of_spatial_transformations * neuron_size * sizeof(float), plan[i].stream);
    }

    // Start kernel
    for (int i = 0; i < number_of_gpus; ++i)
    {
        // Set device
        cudaSetDevice(i);

        // Setup execution parameters
        dim3 dimBlock(block_size);
        dim3 dimGrid(number_of_spatial_transformations, plan[i].size);

        switch (block_size)
        {
            case  512: euclidean_distance_kernel< 512><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotated_images, plan[i].d_first_step, neuron_size); break;
            case  256: euclidean_distance_kernel< 256><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotated_images, plan[i].d_first_step, neuron_size); break;
            case  128: euclidean_distance_kernel< 128><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotated_images, plan[i].d_first_step, neuron_size); break;
            case   64: euclidean_distance_kernel<  64><<<dimGrid, dimBlock, 0, plan[i].stream>>>(
                plan[i].d_som, plan[i].d_rotated_images, plan[i].d_first_step, neuron_size); break;
            default:
                throw pink::exception("generate_euclidean_distance_matrix_first_step: block size not supported");
        }

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch CUDA kernel cuda_generateEuclideanDistanceMatrix_first_step (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        // Copy data
        if (i != 0)
            cudaMemcpyPeerAsync(plan[0].d_first_step + plan[i].offset * number_of_spatial_transformations, 0, plan[i].d_first_step, i, plan[i].size * number_of_spatial_transformations * sizeof(float), plan[i].stream);

        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to cudaMemcpyPeerAsync (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }

    // Shut down GPU devices
    for (int i = 0; i < number_of_gpus; ++i)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(plan[i].stream);
        cudaStreamDestroy(plan[i].stream);
    }

    cudaSetDevice(0);
    cudaDeviceSynchronize();
#endif
}

} // namespace pink
