/*
 * Image.h
 *
 *  Created on: Oct 15, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <Python.h>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace PINK {

//! Rectangular image
template <class T>
class Image
{
public:

	// Parameter constructor
	Image(int height, int width)
     : height_(height), width_(width), pixel_(height*width)
    {}

	//! Write to file in binary mode
	void writeBinary(std::string const& filename);

	//! Show image on screen using python
	void show();

	int getHeight() const { return height_; }
	int getWidth() const { return width_; }
	std::vector<T>& getPixel() { return pixel_; }

private:

	template <class T2>
	friend class ImageIterator;

	int height_;
	int width_;
	std::vector<T> pixel_;

};

template <class T>
void Image<T>::writeBinary(std::string const& filename)
{
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("ImageArray: Error opening " + filename);

    int one(1);
    os.write((char*)&one, sizeof(int));
    os.write((char*)&height_, sizeof(int));
    os.write((char*)&width_, sizeof(int));
    os.write((char*)&pixel_[0], height_ * width_ * sizeof(float));
}

template <class T>
void Image<T>::show()
{
	std::string filename("ImageTmp.bin");
	writeBinary(filename);

    Py_Initialize();
    PyRun_SimpleString("import numpy");
    PyRun_SimpleString("import matplotlib.pylab as plt");
    PyRun_SimpleString("import struct");

    std::string line = "inFile = open(\"" + filename + "\", 'rb')";
    PyRun_SimpleString(line.c_str());
	PyRun_SimpleString("size = struct.unpack('iii', inFile.read(12))");
	PyRun_SimpleString("array = numpy.array(struct.unpack('f'*size[1]*size[2], inFile.read(size[1]*size[2]*4)))");
	PyRun_SimpleString("data = numpy.ndarray([size[1],size[2]], 'float', array)");
	PyRun_SimpleString("inFile.close()");
	PyRun_SimpleString("fig = plt.figure()");
	PyRun_SimpleString("ax = fig.add_subplot(1,1,1)");
	PyRun_SimpleString("ax.set_aspect('equal')");
	PyRun_SimpleString("plt.imshow(data, interpolation='nearest', cmap=plt.cm.ocean)");
	PyRun_SimpleString("plt.colorbar()");
	PyRun_SimpleString("plt.show()");
}

} // namespace PINK

#endif /* IMAGE_H_ */
