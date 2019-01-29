/**
 * @file   CudaLib/rotate_bilinear_kernel.h
 * @date   Jan 28, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cuda_runtime.h>

/// CUDA device kernel for rotating a list of quadratic images
template <typename T>
__global__
void rotate_bilinear_kernel(T const *src, T *dst,
    uint32_t src_height, uint32_t src_width, uint32_t dst_height, uint32_t dst_width,
    float const *cos_alpha, float const *sin_alpha, uint32_t spacing)
{
    const uint32_t dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width or dst_y >= dst_height) return;

    const uint32_t dst_size = dst_width * dst_height;

    const float cos_alpha_local = cos_alpha[blockIdx.z];
    const float sin_alpha_local = sin_alpha[blockIdx.z];

    // Center of src image
    const float src_center_x = (src_width - 1) * 0.5;
    const float src_center_y = (src_height - 1) * 0.5;

    // Center of dst image
    const float dst_center_x = (dst_width - 1) * 0.5;
    const float dst_center_y = (dst_height - 1) * 0.5;

    float dst_position_x = static_cast<float>(dst_x) - dst_center_x;
    float dst_position_y = static_cast<float>(dst_y) - dst_center_y;

    float src_position_x = dst_position_x * cos_alpha_local - dst_position_y * sin_alpha_local + src_center_x;
    float src_position_y = dst_position_x * sin_alpha_local + dst_position_y * cos_alpha_local + src_center_y;

    if (src_position_x < 0.0 or src_position_x > src_width - 1 or
        src_position_y < 0.0 or src_position_y > src_height - 1)
    {
        dst[blockIdx.z * spacing * dst_size + dst_x * dst_height + dst_y] = 0.0;
    }
    else
    {
        int src_x = src_position_x;
        int src_y = src_position_y;

        int src_x_plus_1 = src_x + 1;
        int src_y_plus_1 = src_y + 1;

        float rx = src_position_x - src_x;
        float ry = src_position_y - src_y;

        float cx = 1.0 - rx;
        float cy = 1.0 - ry;

        dst[blockIdx.z * spacing * dst_size + dst_x * dst_height + dst_y] =
              cx * cy * src[src_x * src_height + src_y]
            + cx * ry * src[src_x * src_height + src_y_plus_1]
            + rx * cy * src[src_x_plus_1 * src_height + src_y]
            + rx * ry * src[src_x_plus_1 * src_height + src_y_plus_1];
    }
}
