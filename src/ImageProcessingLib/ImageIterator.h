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
	ImageIterator()
	 : number_(0), count_(0), height_(0), width_(0), ptrStream_(nullptr)
	{}

	//! Parameter constructor
	ImageIterator(std::string const& filename)
	 : number_(0), count_(0), height_(0), width_(0), ptrStream_(new std::ifstream(filename))
	{
	    if (!(*ptrStream_)) throw std::runtime_error("ImageIterator: Error opening " + filename);

	    ptrStream_->read((char*)&number_, sizeof(int));
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

	int number() const { return number_; }

private:

	//! Read next picture
	void next()
	{
		if (count_ < number_) {
		    ptrCurrentImage_ = PtrImage(new ImageType(height_,width_));
	        ptrStream_->read((char*)&ptrCurrentImage_->getPixel()[0], height_ * width_ * sizeof(float));
	        ++count_;
		} else {
			ptrStream_.reset();
		}
	}

	int number_;
	int count_;
	int height_;
	int width_;

	std::shared_ptr<std::ifstream> ptrStream_;

    PtrImage ptrCurrentImage_;

};

} // namespace PINK

#endif /* IMAGEITERATOR_H_ */
