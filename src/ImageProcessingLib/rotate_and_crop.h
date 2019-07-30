/**
 * @file   ImageProcessingLib/rotate_and_crop.h
 * @date   Nov 16, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>
#include <cmath>

#include "UtilitiesLib/Interpolation.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

template <typename T>
void rotate_and_crop_nearest_neighbor(T const* src, T *dst,
    uint32_t src_height, uint32_t src_width, uint32_t dst_height, uint32_t dst_width, float alpha)
{
    const uint32_t width_margin = static_cast<uint32_t>((src_width - dst_width) * 0.5);
    const uint32_t height_margin = static_cast<uint32_t>((src_height - dst_height) * 0.5);

    const float cos_alpha = std::cos(alpha);
    const float sin_alpha = std::sin(alpha);

    const float x0 = (src_width-1) * 0.5f;
    const float y0 = (src_height-1) * 0.5f;
    float x1, y1;

    for (uint32_t x2 = 0; x2 < dst_width; ++x2) {
        for (uint32_t y2 = 0; y2 < dst_height; ++y2) {
            x1 = (static_cast<float>(x2) + width_margin  - x0) * cos_alpha
               + (static_cast<float>(y2) + height_margin - y0) * sin_alpha + x0 + 0.1f;
            if (x1 < 0 or x1 >= src_width) {
                dst[x2*dst_height + y2] = 0.0f;
                continue;
            }
            y1 = (static_cast<float>(y2) + height_margin - y0) * cos_alpha
               - (static_cast<float>(x2) + width_margin  - x0) * sin_alpha + y0 + 0.1f;
            if (y1 < 0 or y1 >= src_height) {
                dst[x2*dst_height + y2] = 0.0f;
                continue;
            }
            dst[x2*dst_height + y2] = src[static_cast<uint32_t>(x1)*src_height + static_cast<uint32_t>(y1)];
        }
    }
}

template <typename T>
void rotate_and_crop_bilinear(T const* src, T *dst,
    uint32_t src_height, uint32_t src_width, uint32_t dst_height, uint32_t dst_width, float alpha)
{
    const uint32_t width_margin = static_cast<uint32_t>((src_width - dst_width) * 0.5);
    const uint32_t height_margin = static_cast<uint32_t>((src_height - dst_height) * 0.5);

    const float cos_alpha = std::cos(alpha);
    const float sin_alpha = std::sin(alpha);

    const float x0 = (src_width - 1) * 0.5f;
    const float y0 = (src_height - 1) * 0.5f;
    float x1, y1, rx1, ry1, cx1, cy1;
    uint32_t ix1, iy1, ix1b, iy1b;

    for (uint32_t x2 = 0; x2 < dst_width; ++x2) {
        for (uint32_t y2 = 0; y2 < dst_height; ++y2) {
            x1 = (static_cast<float>(x2) + width_margin - x0) * cos_alpha
               + (static_cast<float>(y2) + height_margin - y0) * sin_alpha + x0;
//            if (x1 < 0 or x1 >= src_width) {
//                dst[x2*dst_height + y2] = 0.0f;
//                continue;
//            }
            y1 = (static_cast<float>(y2) + height_margin - y0) * cos_alpha
               - (static_cast<float>(x2) + width_margin - x0) * sin_alpha + y0;
//            if (y1 < 0 or y1 >= src_height) {
//                dst[x2*dst_height + y2] = 0.0f;
//                continue;
//            }
            ix1 = static_cast<uint32_t>(x1);
            iy1 = static_cast<uint32_t>(y1);
            ix1b = ix1 + 1;
            iy1b = iy1 + 1;
            rx1 = x1 - ix1;
            ry1 = y1 - iy1;
            cx1 = 1.0f - rx1;
            cy1 = 1.0f - ry1;
            dst[x2 * dst_height + y2] = cx1 * cy1 * src[ix1  * src_height + iy1 ]
                                      + cx1 * ry1 * src[ix1  * src_height + iy1b]
                                      + rx1 * cy1 * src[ix1b * src_height + iy1 ]
                                      + rx1 * ry1 * src[ix1b * src_height + iy1b];
        }
    }
}

template <typename T>
void rotate_and_crop(T const *src, T *dst, uint32_t src_height, uint32_t src_width,
    uint32_t dst_height, uint32_t dst_width, float alpha, Interpolation interpolation)
{
    assert(src_height > 0);
    assert(src_width > 0);
    assert(dst_height > 0);
    assert(dst_width > 0);

    assert(src_height >= dst_height);
    assert(src_width >= dst_width);

    if (interpolation == Interpolation::NEAREST_NEIGHBOR)
        rotate_and_crop_nearest_neighbor(src, dst, src_height, src_width, dst_height, dst_width, alpha);
    else if (interpolation == Interpolation::BILINEAR)
        rotate_and_crop_bilinear(src, dst, src_height, src_width, dst_height, dst_width, alpha);
    else {
        throw pink::exception("rotate_and_crop: unknown interpolation\n");
    }
}

} // namespace pink
