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
void rotate_90_degrees(T *src, T *dst, uint32_t height, uint32_t width)
{
    assert(height > 0);
    assert(width > 0);
    assert(height == width);

    for (uint32_t x = 0; x < width; ++x) {
        for (uint32_t y = 0; y < height; ++y) {
            // TG: I think this actually is counter-clockwise, which
            // TG: throws off the angle in transform by factors of pi
            // TG: radians between pi/2 to pi and 3pi/2 to 2pi
            // dst[(height-y-1)*width + x] = src[x*height + y];
            dst[x*height + y] = src[  (height-y-1)*width + x];
        }
    }
}

} // namespace pink
