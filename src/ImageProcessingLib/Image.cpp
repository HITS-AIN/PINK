/**
 * @file   Image.cpp
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "Image.h"
#include "Python.h"
#include <fstream>

namespace PINK {

template <>
void Image<float>::writeBinary(std::string const& filename)
{
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("ImageArray: Error opening " + filename);

    int one(1);
    os.write((char*)&one, sizeof(int));
    os.write((char*)&height_, sizeof(int));
    os.write((char*)&width_, sizeof(int));
    os.write((char*)&pixel_[0], height_ * width_ * sizeof(float));
}

template <>
void Image<float>::show()
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
