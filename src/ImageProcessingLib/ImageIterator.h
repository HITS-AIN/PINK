/**
 * @file   ImageIterator.h
 * @date   Oct 15, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Image.h"

namespace pink {

//! Read iteratively a binary image file
template <class T>
class ImageIterator
{

    typedef Image<T> ImageType;

public:

    typedef std::shared_ptr<ImageType> PtrImage;

    //! Default constructor
    ImageIterator()
     : numberOfImages_(0), numberOfChannels_(0), count_(0), height_(0), width_(0), ptrStream_(nullptr)
    {}

    //! Parameter constructor
    ImageIterator(std::string const& filename)
     : numberOfImages_(0), numberOfChannels_(0), count_(0), height_(0), width_(0), ptrStream_(new std::ifstream(filename))
    {
        if (!(*ptrStream_)) throw std::runtime_error("ImageIterator: Error opening " + filename);

        // Skip all header lines starting with #
        std::string line;
        int last_position = ptrStream_->tellg();
        while (std::getline(*ptrStream_, line)) {
            if (line[0] != '#') break;
            last_position = ptrStream_->tellg();
        }

        ptrStream_->seekg(last_position, ptrStream_->beg);
        ptrStream_->read((char*)&numberOfImages_, sizeof(int));
        ptrStream_->read((char*)&numberOfChannels_, sizeof(int));
        ptrStream_->read((char*)&height_, sizeof(int));
        ptrStream_->read((char*)&width_, sizeof(int));

        next();
    }

    //! Equal comparison
    bool operator == (ImageIterator const& other) const
    {
        return ptrStream_ == other.ptrStream_;
    }

    //! Unequal comparison
    bool operator != (ImageIterator const& other) const
    {
        return !operator==(other);
    }

    //! Prefix increment
    ImageIterator& operator ++ ()
    {
        next();
        return *this;
    }

    //! Addition assignment operator
    ImageIterator& operator += (int step)
    {
        ptrStream_->seekg((step - 1) * height_ * width_ * sizeof(float), ptrStream_->cur);
        next();
        return *this;
    }

    //! Dereference
    ImageType& operator * () const
    {
        return *ptrCurrentImage_;
    }

    //! Dereference
    ImageType* operator -> () const
    {
        return &(operator*());
    }

    //! Return number of images.
    int getNumberOfImages() const { return numberOfImages_; }

    //! Return number of channels.
    int getNumberOfChannels() const { return numberOfChannels_; }

private:

    //! Read next picture
    void next()
    {
        if (count_ < numberOfImages_) {
            ptrCurrentImage_ = std::make_shared<ImageType>(height_, width_, numberOfChannels_);
            ptrStream_->read((char*)&ptrCurrentImage_->getPixel()[0], numberOfChannels_ * height_ * width_ * sizeof(float));
            ++count_;
        } else {
            ptrStream_.reset();
        }
    }

    int numberOfImages_;
    int numberOfChannels_;
    int count_;
    int height_;
    int width_;

    std::shared_ptr<std::ifstream> ptrStream_;

    PtrImage ptrCurrentImage_;

};

} // namespace pink
