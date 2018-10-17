/**
 * @file   CudaLib/generate_rotated_images_gpu.cu
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdio.h>

#include "crop.h"
#include "CudaLib.h"
#include "flip.h"
#include "rotate_and_crop_nearest_neighbor.h"
#include "rotate_and_crop_bilinear.h"
#include "rotate_90degrees_list.h"

namespace pink {

#define BLOCK_SIZE 32

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void generate_rotated_images_gpu(thrust::device_vector<T> d_rotatedImages, thrust::device_vector<T> d_image,
    int num_rot, int image_dim, int neuron_dim, bool useFlip, Interpolation interpolation,
    thrust::device_vector<T> d_cosAlpha, thrust::device_vector<T> d_sinAlpha, int numberOfChannels)
{
//    int neuron_size = neuron_dim * neuron_dim;
//    int image_size = image_dim * image_dim;
//
//    // Crop first image
//    {
//        // Setup execution parameters
//        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
//        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//        dim3 dimGrid(gridSize, gridSize);
//
//        // Start kernel
//        for (int c = 0; c < numberOfChannels; ++c)
//        {
//            crop<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size,
//                d_image + c*image_size, neuron_dim, image_dim);
//
//            cudaError_t error = cudaGetLastError();
//
//            if (error != cudaSuccess)
//            {
//                fprintf(stderr, "Failed to launch CUDA kernel crop (error code %s)!\n", cudaGetErrorString(error));
//                exit(EXIT_FAILURE);
//            }
//        }
//    }
//
//    if (num_rot == 1) return;
//
//    // Rotate images between 0 and 90 degrees
//    {
//        // Setup execution parameters
//        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
//        int num_real_rot = num_rot/4-1;
//
//        if (num_real_rot) {
//            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//            dim3 dimGrid(gridSize, gridSize, num_real_rot);
//
//            // Start kernel
//            for (int c = 0; c < numberOfChannels; ++c)
//            {
//                if (interpolation == Interpolation::NEAREST_NEIGHBOR)
//                    rotate_and_crop_nearest_neighbor<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + (c+numberOfChannels)*neuron_size, d_image + c*image_size,
//                        neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha, numberOfChannels);
//                else if (interpolation == Interpolation::BILINEAR)
//                    rotate_and_crop_bilinear<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + (c+numberOfChannels)*neuron_size, d_image + c*image_size,
//                        neuron_size, neuron_dim, image_dim, d_cosAlpha, d_sinAlpha, numberOfChannels);
//                else {
//                    fprintf(stderr, "generate_rotated_images_gpu: unknown interpolation type!\n");
//                    exit(EXIT_FAILURE);
//                }
//
//                cudaError_t error = cudaGetLastError();
//
//                if (error != cudaSuccess)
//                {
//                    fprintf(stderr, "Failed to launch CUDA kernel rotateAndCrop (error code %s)!\n", cudaGetErrorString(error));
//                    exit(EXIT_FAILURE);
//                }
//            }
//        }
//    }
//
//    // Special 90 degree rotation for remaining rotations between 90 and 360 degrees
//    {
//        // Setup execution parameters
//        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
//        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//        dim3 dimGrid(gridSize, gridSize, num_rot/4);
//
//        int offset = num_rot/4 * numberOfChannels * neuron_size;
//        int mc_neuron_size = numberOfChannels * neuron_size;
//
//        // Start kernel
//        for (int c = 0; c < numberOfChannels; ++c)
//        {
//            rotate_90degrees_list<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size,
//                neuron_dim, mc_neuron_size, offset);
//            rotate_90degrees_list<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size + offset,
//                neuron_dim, mc_neuron_size, offset);
//            rotate_90degrees_list<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + c*neuron_size + 2*offset,
//                neuron_dim, mc_neuron_size, offset);
//
//            cudaError_t error = cudaGetLastError();
//
//            if (error != cudaSuccess)
//            {
//                fprintf(stderr, "Failed to launch CUDA kernel rotate_90degrees_list (error code %s)!\n", cudaGetErrorString(error));
//                exit(EXIT_FAILURE);
//            }
//        }
//    }
//
//    if (useFlip)
//    {
//        // Setup execution parameters
//        int gridSize = ceil((float)neuron_dim/BLOCK_SIZE);
//        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//        dim3 dimGrid(gridSize, gridSize, num_rot * numberOfChannels);
//
//        // Start kernel
//        for (int c = 0; c < numberOfChannels; ++c)
//        {
//            flip<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_rotatedImages + num_rot * numberOfChannels * neuron_size,
//                d_rotatedImages, neuron_dim, neuron_size);
//
//            cudaError_t error = cudaGetLastError();
//
//            if (error != cudaSuccess)
//            {
//                fprintf(stderr, "Failed to launch CUDA kernel flip (error code %s)!\n", cudaGetErrorString(error));
//                exit(EXIT_FAILURE);
//            }
//        }
//    }
}

template
void generate_rotated_images_gpu<float>(thrust::device_vector<float> d_rotatedImages, thrust::device_vector<float> d_image,
    int num_rot, int image_dim, int neuron_dim, bool useFlip, Interpolation interpolation,
    thrust::device_vector<float> d_cosAlpha, thrust::device_vector<float> d_sinAlpha, int numberOfChannels);

} // namespace pink
