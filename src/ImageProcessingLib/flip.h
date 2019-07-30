/**
 * @file   ImageProcessingLib/flip.h
 * @date   Oct 31, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cassert>

namespace pink {

template <typename T>
void flip(T const *src, T *dst, uint32_t height, uint32_t width)
{
    assert(height > 0);
    assert(width > 0);
    assert(height == width);

    T *pdst = dst + (height-1) * width;
    T const *psrc = src;

    for (uint32_t i = 0; i < height; ++i) {
        for (uint32_t j = 0; j < width; ++j) {
            pdst[j] = psrc[j];
        }
        pdst -= width;
        psrc += width;
    }
}

} // namespace pink
