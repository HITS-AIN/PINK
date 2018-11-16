/**
 * @file   CudaLib/rotate_and_crop_bilinear_kernel.h
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code for combined rotation and cropping of a list of quadratic images.
 */
template <typename T>
__global__
void rotate_and_crop_bilinear_kernel(T *rotated_images, T const *image, uint32_t neuron_size,
    uint32_t neuron_dim, uint32_t image_dim, float const *cos_alpha, float const *sin_alpha, uint32_t spacing)
{
    uint32_t x2 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (x2 >= neuron_dim or y2 >= neuron_dim) return;

    T center = (image_dim - 1) * 0.5;
    T margin = (image_dim - neuron_dim) * 0.5;
    T center_margin = center - margin;

    T cos_alpha_local = cos_alpha[blockIdx.z];
    T sin_alpha_local = sin_alpha[blockIdx.z];

    T x1 = (x2-center_margin)*cos_alpha_local + (y2-center_margin)*sin_alpha_local + center + 0.1;
    T y1 = (y2-center_margin)*cos_alpha_local - (x2-center_margin)*sin_alpha_local + center + 0.1;

    uint32_t ix1 = x1;
    uint32_t iy1 = y1;
    uint32_t ix1b = ix1 + 1;
    uint32_t iy1b = iy1 + 1;

    T rx1 = x1 - ix1;
    T ry1 = y1 - iy1;
    T cx1 = 1.0f - rx1;
    T cy1 = 1.0f - ry1;

    T* pCurRot = rotated_images + blockIdx.z * spacing * neuron_size;

    T value = cx1 * cy1 * image[ix1  * image_dim + iy1 ]
            + cx1 * ry1 * image[ix1  * image_dim + iy1b]
            + rx1 * cy1 * image[ix1b * image_dim + iy1 ]
            + rx1 * ry1 * image[ix1b * image_dim + iy1b];

    if (x1 >= 0 and x1 < image_dim and y1 >= 0 and y1 < image_dim) {
        atomicExch(pCurRot + x2*neuron_dim + y2, value);
    } else {
        atomicExch(pCurRot + x2*neuron_dim + y2, 0.0f);
    }
}
