/**
 * @file   CudaLib/resize_kernel.h
 * @date   Jan 28, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cuda_runtime.h>

namespace pink {

/// CUDA device kernel to resize an quadratic, row-major image
template <typename T>
__global__
void resize_kernel(T *dst, T const *src, uint32_t dst_dim, uint32_t src_dim, uint32_t min_dim)
{
    assert(min_dim == std::min(dst_dim, src_dim));

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= min_dim or j >= min_dim) return;

    uint32_t src_margin = 0, dst_margin = 0;
    if (src_dim < dst_dim) dst_margin = static_cast<uint32_t>((dst_dim - src_dim) * 0.5);
    else if (src_dim > dst_dim) src_margin = static_cast<uint32_t>((src_dim - dst_dim) * 0.5);

    dst[(i + dst_margin) * dst_dim + (j + dst_margin)] =
    src[(i + src_margin) * src_dim + (j + src_margin)];
}

} // namespace pink
