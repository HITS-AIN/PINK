/**
 * @file   CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <limits>
#include <omp.h>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector>

#include "euclidean_distance_kernel.h"

namespace pink {

/// Multiply all vector elements with scalar
template <typename T>
std::vector<T> operator * (std::vector<T> const& v, T scalar)
{
    std::vector<T> r(v);
    for (auto& e : r) e *= scalar;
    return r;
}

template <typename T>
std::vector<thrust::device_vector<T>> allocate_local_memory(std::vector<uint32_t> const& sizes)
{
    std::vector<thrust::device_vector<T>> result(sizes.size());

    auto&& gpu_ids = cuda_get_gpu_ids();
    for (uint32_t i = 0; i < sizes.size(); ++i)
    {
        cudaSetDevice(gpu_ids[i + 1]);
        result[i].resize(sizes[i]);
    }
    cudaSetDevice(gpu_ids[0]);

    return result;
}

/// Calculate euclidean distance on multiple GPU devices
template <typename DataType, typename EuclideanType>
void generate_euclidean_distance_matrix_first_step_multi_gpu(thrust::device_vector<EuclideanType> const& d_som,
    thrust::device_vector<EuclideanType> const& d_rotated_images, thrust::device_vector<DataType>& d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size, uint32_t neuron_size, uint32_t block_size)
{
    auto&& gpu_ids = cuda_get_gpu_ids();
    auto number_of_gpus = gpu_ids.size();
    auto number_of_threads = static_cast<uint32_t>(omp_get_max_threads());

    if (number_of_threads < number_of_gpus) {
        std::cout << "Number of threads = " << number_of_threads << std::endl;
        std::cout << "Number of GPUs = " << number_of_gpus << std::endl;
        std::cout << "GPU IDs = ";
        for (auto id : gpu_ids) std::cout << id << " ";
        std::cout << std::endl;
        throw pink::exception("Number of CPU threads must not be smaller than the number of GPU devices");
    }

    // Set size
    std::vector<uint32_t> size(number_of_gpus);
    uint32_t rest = som_size % number_of_gpus;
    for (uint32_t i = 0; i < number_of_gpus; ++i) {
        size[i] = som_size / number_of_gpus;
        if (rest > i) ++size[i];
    }

    // Set offset
    std::vector<uint32_t> offset(number_of_gpus);
    offset[0] = 0;
    for (uint32_t i = 1; i < number_of_gpus; ++i) {
        offset[i] = offset[i-1] + size[i-1];
    }

    static auto d_som_local = allocate_local_memory<EuclideanType>(
        std::vector<uint32_t>(size.begin() + 1, size.end()) * neuron_size);
    static auto d_rotated_images_local = allocate_local_memory<EuclideanType>(
        std::vector<uint32_t>(number_of_gpus - 1, number_of_spatial_transformations * neuron_size));
    static auto d_first_step_local = allocate_local_memory<DataType>(
        std::vector<uint32_t>(size.begin() + 1, size.end()) * number_of_spatial_transformations);

    assert(number_of_gpus < std::numeric_limits<int>::max());
    for (int i = 1; i < static_cast<int>(number_of_gpus); ++i)
    {
        auto si = static_cast<size_t>(i);

        // Set GPU device
        cudaSetDevice(gpu_ids[si]);

        // Copy data
        gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_som_local[si - 1].data()), i,
                            thrust::raw_pointer_cast(d_som.data()) + offset[si] * neuron_size, 0,
                            size[si] * neuron_size * sizeof(EuclideanType)));

        gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_rotated_images_local[si - 1].data()), i,
                            thrust::raw_pointer_cast(d_rotated_images.data()), 0,
                            number_of_spatial_transformations * neuron_size * sizeof(EuclideanType)));

        gpuErrchk(cudaDeviceSynchronize());
    }

    std::vector<std::thread> workers;
    for (int i = 1; i < static_cast<int>(number_of_gpus); ++i)
    {
        auto si = static_cast<size_t>(i);

        workers.push_back(std::thread([&, si]()
        {
            // Set GPU device
            cudaSetDevice(gpu_ids[si]);

            // Setup execution parameters
            dim3 dim_block(block_size);
            dim3 dim_grid(number_of_spatial_transformations, size[si]);

            switch (block_size)
            {
                case  512: euclidean_distance_kernel< 512><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_rotated_images_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_first_step_local[si - 1].data()), neuron_size);
                    break;
                case  256: euclidean_distance_kernel< 256><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_rotated_images_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_first_step_local[si - 1].data()), neuron_size);
                    break;
                case  128: euclidean_distance_kernel< 128><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_rotated_images_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_first_step_local[si - 1].data()), neuron_size);
                    break;
                case   64: euclidean_distance_kernel<  64><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_rotated_images_local[si - 1].data()),
                    thrust::raw_pointer_cast(d_first_step_local[si - 1].data()), neuron_size);
                    break;
                default:
                    throw pink::exception("generate_euclidean_distance_matrix_first_step: block size not supported");
            }
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }));
    }

    // Set GPU device to master
    cudaSetDevice(gpu_ids[0]);

    // Setup execution parameters
    dim3 dim_block(block_size);
    dim3 dim_grid(number_of_spatial_transformations, size[0]);

    switch (block_size)
    {
        case 512: euclidean_distance_kernel<512><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som.data()), thrust::raw_pointer_cast(d_rotated_images.data()),
            thrust::raw_pointer_cast(d_first_step.data()), neuron_size);
            break;
        case 256: euclidean_distance_kernel<256><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som.data()), thrust::raw_pointer_cast(d_rotated_images.data()),
            thrust::raw_pointer_cast(d_first_step.data()), neuron_size);
            break;
        case 128: euclidean_distance_kernel<128><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som.data()), thrust::raw_pointer_cast(d_rotated_images.data()),
            thrust::raw_pointer_cast(d_first_step.data()), neuron_size);
            break;
        case 64: euclidean_distance_kernel<64><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som.data()), thrust::raw_pointer_cast(d_rotated_images.data()),
            thrust::raw_pointer_cast(d_first_step.data()), neuron_size);
            break;
        default:
            throw pink::exception("generate_euclidean_distance_matrix_first_step: block size not supported");
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Wait for all workers
    for (auto&& w : workers) w.join();

    for (int i = 1; i < static_cast<int>(number_of_gpus); ++i)
    {
        auto si = static_cast<size_t>(i);

        // Copy data
        gpuErrchk(cudaMemcpyPeer(
            thrust::raw_pointer_cast(d_first_step.data()) + offset[si] * number_of_spatial_transformations, 0,
            thrust::raw_pointer_cast(d_first_step_local[si - 1].data()),
            i, size[si] * number_of_spatial_transformations * sizeof(DataType)));
    }

    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace pink
