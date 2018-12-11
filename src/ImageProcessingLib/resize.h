/**
 * @file   ImageProcessingLib/resize.h
 * @date   Oct 31, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

/// row-major
template <typename T>
void resize(T const* src, T *dst, int src_height, int src_width, int dst_height, int dst_width)
{
    int src_width_margin = 0, dst_width_margin = 0;
    if (src_width < dst_width) dst_width_margin = (dst_width - src_width) / 2;
    else if (src_width > dst_width) src_width_margin = (src_width - dst_width) / 2;

    int src_height_margin = 0, dst_height_margin = 0;
    if (src_height < dst_height) dst_height_margin = (dst_height - src_height) / 2;
    else if (src_height > dst_height) src_height_margin = (src_height - dst_height) / 2;

    int width = std::min(src_width, dst_width);
    int height = std::min(src_height, dst_height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            dst[(i+dst_height_margin) * dst_width + (j+dst_width_margin)] =
            src[(i+src_height_margin) * src_width + (j+src_width_margin)];
        }
    }
}

} // namespace pink
