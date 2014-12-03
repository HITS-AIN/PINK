/**
 * @file   Image.cpp
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "Image.h"
#include "ImageProcessing.h"
#include <fstream>

namespace PINK {

template <>
void Image<float>::writeBinary(std::string const& filename)
{
	writeImagesToBinaryFile(pixel_, 1, 1, height_, width_, filename);
}

template <>
void Image<float>::show()
{
	showImage(pixel_, height_, width_);
}

} // namespace PINK
