/**
 * @file   Image.cpp
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <fstream>

#include "Image.h"
#include "ImageProcessing.h"

namespace pink {

template <>
void Image<float>::writeBinary(std::string const& filename)
{
    writeImagesToBinaryFile(pixel_, 1, 1, height_, width_, filename);
}

} // namespace pink
