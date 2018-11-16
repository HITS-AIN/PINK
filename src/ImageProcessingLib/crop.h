/**
 * @file   ImageProcessingLib/crop.h
 * @date   Nov 16, 2018
 * @author Bernd Doser, HITS gGmbH
 */

namespace pink {

/// column major
template <typename T>
void crop(T const* src, T *dst, int src_height, int src_width, int dst_height, int dst_width)
{
    int width_margin = (src_width - dst_width) / 2;
    int height_margin = (src_height - dst_height) / 2;

    for (int i = 0; i < dst_height; ++i) {
        for (int j = 0; j < dst_width; ++j) {
            dst[i*dst_width+j] = src[(i+height_margin)*src_width + (j+width_margin)];
        }
    }
}

} // namespace pink
