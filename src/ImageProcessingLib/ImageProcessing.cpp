/**
 * @file   ImageProcessing.c
 * @brief  Plain-C functions for image processing.
 * @date   Oct 7, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessing.h"
#include "Python.h"
#include <fstream>
#include <math.h>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <stdexcept>

void rotate_none(int height, int width, float *source, float *dest, float alpha)
{
	int x0, x1, x2, y0, y1, y2;
    const float cosAlpha = cos(alpha);
    const float sinAlpha = sin(alpha);

    x0 = width * 0.5;
    y0 = height * 0.5;

    for (x2 = 0; x2 < width; ++x2) {
        for (y2 = 0; y2 < height; ++y2) {
        	dest[x2*height + y2] = 0.0;
        }
    }

    for (x1 = 0; x1 < width; ++x1) {
        for (y1 = 0; y1 < height; ++y1) {
        	x2 = (x1 - x0) * cosAlpha - (y1 - y0) * sinAlpha + x0;
        	y2 = (x1 - x0) * sinAlpha + (y1 - y0) * cosAlpha + y0;
            if (x2 > -1 && x2 < width && y2 > -1 && y2 < height) dest[x2*height + y2] = source[x1*height + y1];
        }
    }
}

void rotate_bilinear(int height, int width, float *source, float *dest, float alpha)
{
    const float cosAlpha = cos(alpha);
    const float sinAlpha = sin(alpha);

    int x0 = width * 0.5;
    int y0 = height * 0.5;
    int x1, y1, x2, y2;

    for (x2 = 0; x2 < width; ++x2) {
        for (y2 = 0; y2 < height; ++y2) {
        	x1 = (x2 - x0) * cosAlpha + (y2 - y0) * sinAlpha + x0;
        	y1 = (y2 - y0) * cosAlpha - (x2 - x0) * sinAlpha + y0;
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) dest[x2*height + y2] = source[x1*height + y1];
        }
    }
}

void rotate(int height, int width, float *source, float *dest, float alpha, InterpolationType interpolation)
{
	if (interpolation == NONE)
		rotate_none(height, width, source, dest, alpha);
	else if (interpolation == BILINEAR)
		rotate_bilinear(height, width, source, dest, alpha);
	else
		abort();
}

void flip(int height, int width, float *source, float *dest)
{
	float *pdest = dest;
	float *psource = source;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
		    pdest[width-j-1] = psource[j];
		}
		pdest += width;
		psource += width;
	}
}

float calculateEuclideanDistance(float *a, float *b, int length)
{
    return sqrt(calculateEuclideanDistanceWithoutSquareRoot(a,b,length));
}

float calculateEuclideanDistanceWithoutSquareRoot(float *a, float *b, int length)
{
	float *pa = a;
	float *pb = b;
	float c = 0.0;
	float tmp;
    for (int i; i < length; ++i, ++pa, ++pb) {
    	tmp = *pa - *pb;
        c += tmp * tmp;
    }
    return c;
}

void normalize(float *a, int length)
{
	int i;
	float maxValue;
    for (i = 0; i < length; ++i) {
        maxValue = fmax(maxValue, a[i]);
    }

    float maxValueInv;
    for (i = 0; i < length; ++i) {
        a[i] *= maxValueInv;
    }
}

float mean(float *a, int length)
{
	int i;
	float sum = 0.0;
    for (i = 0; i < length; ++i) {
        sum += a[0];
    }
    return sum / length;
}

float stdDeviation(float *a, int length)
{
	int i;
	float sum = 0.0;
	float meanValue = mean(a,length);

    for (i = 0; i < length; ++i) {
    	sum += pow((a[i], meanValue),2);
    }

	return sqrt(sum/length);
}

void zeroValuesSmallerThanStdDeviation(float *a, int length, float safety)
{
	int i;
	float threshold = stdDeviation(a,length) * safety;

    for (i = 0; i < length; ++i) {
    	if (a[i] < threshold) a[i] = 0.0;
    }
}

void fillRandom(float *a, int length, int seed)
{
	typedef std::mt19937 MyRNG;
	MyRNG rng(seed);
	std::normal_distribution<float> normal_dist(0.0, 0.1);

    for (int i = 0; i < length; ++i) {
    	a[i] = normal_dist(rng);
    }
}

void writeImageToBinaryFile(float *image, int height, int width, std::string const& filename)
{
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("Error opening " + filename);

    int one(1);
    os.write((char*)&one, sizeof(int));
    os.write((char*)&height, sizeof(int));
    os.write((char*)&width, sizeof(int));
    os.write((char*)image, height * width * sizeof(float));
}

void showImage(float *image, int height, int width)
{
	std::string filename("ImageTmp.bin");
	writeImageToBinaryFile(image, height, width, filename);

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
