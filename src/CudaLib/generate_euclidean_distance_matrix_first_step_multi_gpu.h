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
    int number_of_threads = omp_get_num_threads();

    if (number_of_threads < number_of_gpus)
        throw pink::exception("Number of CPU threads must not be smaller than the number of GPU devices");

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
    for (int i = 0; i < number_of_gpus; ++i) {
    	workers.push_back(std::thread([&]()
    	{
    		std::cout << "Worker i says hello" << std::endl;

    	    // Start GPU device
    	    cudaSetDevice(gpu_ids[i]);
    	    cudaStream_t stream;
    	    cudaStreamCreate(&stream);

#if 0
            // Allocate device memory
            thrust::device_vector<EuclideanType> d_som(size[i] * neuron_size);
            thrust::device_vector<EuclideanType> d_rotated_images(number_of_spatial_transformations * neuron_size);
            thrust::device_vector<DataType> d_first_step(size[i] * number_of_spatial_transformations);

            // Copy data
            cudaMemcpyPeerAsync(plan[i].d_som, i, plan[0].d_som + plan[i].offset * neuron_size, 0, plan[i].size * neuron_size * sizeof(float), plan[i].stream);
            cudaMemcpyPeerAsync(plan[i].d_rotated_images, i, plan[0].d_rotated_images, 0, number_of_spatial_transformations * neuron_size * sizeof(float), plan[i].stream);

    	    // Setup execution parameters
			dim3 dimBlock(block_size);
			dim3 dimGrid(number_of_spatial_transformations, size[i]);

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
#endif

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
