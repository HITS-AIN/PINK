/**
 * @file   ImageIterator.h
 * @date   Oct 15, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef IMAGEITERATOR_H_
#define IMAGEITERATOR_H_

#include "Image.h"
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace PINK {

//! Read iteratively a binary file
template <class T>
class ImageIterator
{

	typedef Image<T> ImageType;

public:

	typedef std::shared_ptr<ImageType> PtrImage;

	//! Default constructor
	ImageIterator();

	//! Parameter constructor
	ImageIterator(std::string const& filename)
	 : stream_(filename)
	{
	    if (!stream_) throw std::runtime_error("ImageIterator: Error opening " + filename);

	    stream_.read((char*)&number_, sizeof(int));
	    stream_.read((char*)&height_, sizeof(int));
	    stream_.read((char*)&width_, sizeof(int));

	    next();
	}

	//! Comparison
	bool operator == (ImageIterator const& other) const
	{
        return stream_ == other.stream_;
	}

	//! Prefix increment
	ImageIterator& operator ++ ()
	{
	    next();
	    return *this;
	}

	//! Dereference
	PtrImage operator * () const
	{
        return ptrCurrentImage_;
	}

private:

	//! Read next picture
	void next()
	{
		ptrCurrentImage_ = PtrImage(new ImageType(height_,width_));
	    stream_.read((char*)&ptrCurrentImage_->pixel_[0], height_ * width_ * sizeof(float));
	}

	int number_;
	int height_;
	int width_;

    std::ifstream stream_;

    PtrImage ptrCurrentImage_;

};

} // namespace PINK

#endif /* IMAGEITERATOR_H_ */
