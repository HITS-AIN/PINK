/**
 * @file   CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <omp.h>
#include <thread>
#include <thrust/device_vector.h>
#include <vector>

#include "euclidean_distance_kernel.h"

namespace pink {

/// Calculate euclidean distance on multiple GPU devices
template <typename DataType, typename EuclideanType>
void generate_euclidean_distance_matrix_first_step_multi_gpu(thrust::device_vector<EuclideanType> const& d_som,
    thrust::device_vector<EuclideanType> const& d_rotated_images, thrust::device_vector<DataType> d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size, uint32_t neuron_size, uint16_t block_size)
{
    auto&& gpu_ids = cuda_get_gpu_ids();
    int number_of_gpus = cuda_get_gpu_ids().size();
    int number_of_threads = omp_get_max_threads();

    if (number_of_threads < number_of_gpus) {
    	std::cout << "Number of threads = " << number_of_threads << std::endl;
    	std::cout << "Number of GPUs = " << number_of_gpus << std::endl;
    	std::cout << "GPU IDs = ";
    	for (auto id : cuda_get_gpu_ids()) std::cout << id << " ";
    	std::cout << std::endl;
        throw pink::exception("Number of CPU threads must not be smaller than the number of GPU devices");
    }

    // Set size
    std::vector<int> size(number_of_gpus);
    int rest = som_size % number_of_gpus;
    for (int i = 0; i < number_of_gpus; ++i) {
        size[i] = som_size / number_of_gpus;
        if (rest > i) ++size[i];
    }

    // Set offset
    std::vector<int> offset(number_of_gpus);
    offset[0] = 0;
    for (int i = 1; i < number_of_gpus; ++i) {
        offset[i] = offset[i-1] + size[i-1];
    }

    std::vector<std::thread> workers;
    for (int i = 1; i < number_of_gpus; ++i) {
        workers.push_back(std::thread([&]()
        {
            std::cout << "Worker << " << i << " says hello" << std::endl;

            // Start GPU device
            cudaSetDevice(gpu_ids[i]);
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            // Allocate device memory
            thrust::device_vector<EuclideanType> d_som_local(size[i] * neuron_size);
            thrust::device_vector<EuclideanType> d_rotated_images_local(number_of_spatial_transformations * neuron_size);
            thrust::device_vector<DataType> d_first_step_local(size[i] * number_of_spatial_transformations);

            // Copy data
            cudaMemcpyPeerAsync(thrust::raw_pointer_cast(d_som_local.data()), i,
                                thrust::raw_pointer_cast(d_som.data()) + offset[i] * neuron_size, 0,
                                size[i] * neuron_size * sizeof(EuclideanType), stream);
            cudaMemcpyPeerAsync(thrust::raw_pointer_cast(d_rotated_images_local.data()), i,
                                thrust::raw_pointer_cast(d_rotated_images.data()), 0,
                                number_of_spatial_transformations * neuron_size * sizeof(EuclideanType), stream);

            // Setup execution parameters
            dim3 dimBlock(block_size);
            dim3 dimGrid(number_of_spatial_transformations, size[i]);

            switch (block_size)
            {
                case  512: euclidean_distance_kernel< 512><<<dimGrid, dimBlock, 0, stream>>>(
                    thrust::raw_pointer_cast(d_som_local.data()), thrust::raw_pointer_cast(d_rotated_images_local.data()),
                    thrust::raw_pointer_cast(d_first_step_local.data()), neuron_size); break;
                case  256: euclidean_distance_kernel< 256><<<dimGrid, dimBlock, 0, stream>>>(
                    thrust::raw_pointer_cast(d_som_local.data()), thrust::raw_pointer_cast(d_rotated_images_local.data()),
                    thrust::raw_pointer_cast(d_first_step_local.data()), neuron_size); break;
                case  128: euclidean_distance_kernel< 128><<<dimGrid, dimBlock, 0, stream>>>(
                    thrust::raw_pointer_cast(d_som_local.data()), thrust::raw_pointer_cast(d_rotated_images_local.data()),
                    thrust::raw_pointer_cast(d_first_step_local.data()), neuron_size); break;
                case   64: euclidean_distance_kernel<  64><<<dimGrid, dimBlock, 0, stream>>>(
                    thrust::raw_pointer_cast(d_som_local.data()), thrust::raw_pointer_cast(d_rotated_images_local.data()),
                    thrust::raw_pointer_cast(d_first_step_local.data()), neuron_size); break;
                default:
                    throw pink::exception("generate_euclidean_distance_matrix_first_step: block size not supported");
            }
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Copy data
            gpuErrchk(cudaMemcpyPeerAsync(thrust::raw_pointer_cast(d_first_step.data()) + offset[i] * number_of_spatial_transformations, 0,
                thrust::raw_pointer_cast(d_first_step_local.data()), i, size[i] * number_of_spatial_transformations * sizeof(DataType), stream));

            // Shut down GPU device
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }));
    }

    // Wait for all workers
    for (auto&& w : workers) w.join();

    cudaSetDevice(gpu_ids[0]);
    cudaDeviceSynchronize();
}

} // namespace pink
