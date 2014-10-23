/**
 * @file   Image.h
 * @date   Oct 15, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <stdexcept>
#include <vector>

namespace PINK {

//! Rectangular image
template <class T>
class Image
{
public:

	//! Parameter constructor, pixel not initialized
	Image(int height, int width)
     : height_(height), width_(width), pixel_(height*width)
    {}

	//! Parameter constructor, pixel initialized
	Image(int height, int width, T t)
     : height_(height), width_(width), pixel_(height*width,t)
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

	//! Show image on screen using python
	void show();

	int getHeight() const { return height_; }
	int getWidth() const { return width_; }
	std::vector<T>& getPixel() { return pixel_; }
	T* getPointerOfFirstPixel() { return &pixel_[0]; }

private:

	//template <class T2>
	//friend class ImageIterator;

	int height_;
	int width_;
	std::vector<T> pixel_;

};

template <>
void Image<float>::writeBinary(std::string const& filename);

template <>
void Image<float>::show();

} // namespace PINK

#endif /* IMAGE_H_ */
