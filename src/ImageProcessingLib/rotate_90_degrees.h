/**
 * @file   ImageProcessingLib/rotate_90_degrees.h
 * @date   Oct 31, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>

namespace pink {

/// Rotate quadratic image clockwise by 90 degrees
template <typename T>
void rotate_90_degrees(T *src, T *dst, int height, int width)
{
    assert(height > 0);
    assert(width > 0);
    assert(height == width);

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            dst[(height-y-1)*width + x] = src[x*height + y];
        }
    }
}

} // namespace pink
