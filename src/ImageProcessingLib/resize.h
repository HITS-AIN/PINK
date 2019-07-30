/**
 * @file   ImageProcessingLib/resize.h
 * @date   Oct 31, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>

namespace pink {

/// row-major
template <typename T>
void resize(T const* src, T *dst, uint32_t src_height, uint32_t src_width,
    uint32_t dst_height, uint32_t dst_width)
{
    assert(src_height > 0);
    assert(src_width > 0);
    assert(dst_height > 0);
    assert(dst_width > 0);

    uint32_t src_width_margin = 0, dst_width_margin = 0;
    if (src_width < dst_width) dst_width_margin = (dst_width - src_width) / 2;
    else if (src_width > dst_width) src_width_margin = (src_width - dst_width) / 2;

    uint32_t src_height_margin = 0, dst_height_margin = 0;
    if (src_height < dst_height) dst_height_margin = (dst_height - src_height) / 2;
    else if (src_height > dst_height) src_height_margin = (src_height - dst_height) / 2;

    uint32_t width = std::min(src_width, dst_width);
    uint32_t height = std::min(src_height, dst_height);

    for (uint32_t i = 0; i < height; ++i) {
        for (uint32_t j = 0; j < width; ++j) {
            dst[(i+dst_height_margin) * dst_width + (j+dst_width_margin)] =
            src[(i+src_height_margin) * src_width + (j+src_width_margin)];
        }
    }
}

} // namespace pink
