/**
 * @file   ImageProcessingLib/Image.h
 * @date   Oct 15, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stdexcept>
#include <vector>

namespace pink {

//! Rectangular image
template <class T>
class Image
{
public:

    //! Parameter constructor, pixel not initialized
    Image(int height, int width, int numberOfChannels = 1)
     : height_(height), width_(width), numberOfChannels_(numberOfChannels), pixel_(numberOfChannels*height*width)
    {}

    //! Parameter constructor, pixel initialized
    Image(int height, int width, int numberOfChannels, T t)
     : height_(height), width_(width), numberOfChannels_(numberOfChannels), pixel_(numberOfChannels*height*width, t)
    {}

    //! Equal comparison
    bool operator == (Image const& other) const
    {
        return height_ == other.height_
            and width_ == other.width_
            and pixel_ == other.pixel_;
    }

    //! Unequal Comparison
    bool operator != (Image const& other) const
    {
        return !operator==(other);
    }

    //! Write to file in binary mode
    void writeBinary(std::string const& filename);

    int getHeight() const { return height_; }
    int getWidth() const { return width_; }
    int getNumberOfChannels() const { return numberOfChannels_; }
    int getSize() const { return pixel_.size(); }

    std::vector<T>& getPixel() { return pixel_; }
    T* getPointerOfFirstPixel() { return &pixel_[0]; }

private:

    //template <class T2>
    //friend class ImageIterator;

    int height_;
    int width_;
    int numberOfChannels_;

    std::vector<T> pixel_;

};

//! Template specialization of @writeBinary for float
template <>
void Image<float>::writeBinary(std::string const& filename);

} // namespace pink
