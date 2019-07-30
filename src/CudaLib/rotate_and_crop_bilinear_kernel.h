/**
 * @file   CudaLib/rotate_and_crop_bilinear_kernel.h
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code for combined rotation and cropping of a list of quadratic images.
 */
template <typename T>
__global__
void rotate_and_crop_bilinear_kernel(T *rotated_images, T const *image,
    uint32_t neuron_size, uint32_t neuron_dim, uint32_t image_dim, float const *cos_alpha,
    float const *sin_alpha, uint32_t spacing)
{
    assert(image_dim >= neuron_dim);

    const uint32_t x2 = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (x2 >= neuron_dim or y2 >= neuron_dim) return;

    float center = (image_dim - 1) * 0.5;
    float margin = (image_dim - neuron_dim) * 0.5;
    float center_margin = center - margin;

    float cos_alpha_local = cos_alpha[blockIdx.z];
    float sin_alpha_local = sin_alpha[blockIdx.z];

    float x1 = (x2 - center_margin) * cos_alpha_local
             + (y2 - center_margin) * sin_alpha_local + center + 0.1;
    float y1 = (y2 - center_margin) * cos_alpha_local
             - (x2 - center_margin) * sin_alpha_local + center + 0.1;

    if (x1 >= 0 and x1 < image_dim and y1 >= 0 and y1 < image_dim)
    {
        uint32_t ix1 = x1;
        uint32_t iy1 = y1;
        uint32_t ix1b = ix1 + 1;
        uint32_t iy1b = iy1 + 1;

        float rx1 = x1 - ix1;
        float ry1 = y1 - iy1;
        float cx1 = 1.0f - rx1;
        float cy1 = 1.0f - ry1;

        rotated_images[blockIdx.z * spacing * neuron_size + x2 * neuron_dim + y2] =
              cx1 * cy1 * image[ix1  * image_dim + iy1 ]
            + cx1 * ry1 * image[ix1  * image_dim + iy1b]
            + rx1 * cy1 * image[ix1b * image_dim + iy1 ]
            + rx1 * ry1 * image[ix1b * image_dim + iy1b];
    }
    else
    {
        rotated_images[blockIdx.z * spacing * neuron_size + x2 * neuron_dim + y2] = 0;
    }
}
