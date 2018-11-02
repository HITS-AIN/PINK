/**
 * @file   ImageProcessingLib/rotate_90_degrees.h
 * @date   Oct 31, 2018
 * @author Bernd Doser, HITS gGmbH
 */

namespace pink {

template <typename T>
void rotate_90_degrees(T *src, T *dst, int height, int width)
{
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            dst[(height-y-1)*width + x] = src[x*height + y];
        }
    }
}

} // namespace pink
