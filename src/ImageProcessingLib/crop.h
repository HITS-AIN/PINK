/**
 * @file   ImageProcessingLib/crop.h
 * @date   Nov 16, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>

namespace pink {

/// column major
template <typename T>
void crop(T const* src, T *dst, uint32_t src_height, uint32_t src_width,
    uint32_t dst_height, uint32_t dst_width)
{
    assert(src_height > 0);
    assert(src_width > 0);
    assert(dst_height > 0);
    assert(dst_width > 0);

    assert(src_height >= dst_height);
    assert(src_width >= dst_width);

    uint32_t width_margin = (src_width - dst_width) / 2;
    uint32_t height_margin = (src_height - dst_height) / 2;

    for (uint32_t i = 0; i < dst_height; ++i) {
        for (uint32_t j = 0; j < dst_width; ++j) {
            dst[i*dst_width+j] = src[(i+height_margin)*src_width + (j+width_margin)];
        }
    }
}

} // namespace pink
