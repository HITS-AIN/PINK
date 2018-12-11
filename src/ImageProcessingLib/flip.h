/**
 * @file   ImageProcessingLib/flip.h
 * @date   Oct 31, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

template <typename T>
void flip(T const *src, T *dst, int height, int width)
{
    T *pdst = dst + (height-1) * width;
    T const *psrc = src;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            pdst[j] = psrc[j];
        }
        pdst -= width;
        psrc += width;
    }
}

} // namespace pink
