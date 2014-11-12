/**
 * @file   ImageProcessing.c
 * @brief  Plain-C functions for image processing.
 * @date   Oct 7, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessing.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <stdexcept>

#if PINK_USE_PYTHON
    #include "Python.h"
#endif

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
            if (x1 < 0 or x1 >= width) {
            	dest[x2*height + y2] = 0.0;
            	continue;
            }
        	y1 = (y2 - y0) * cosAlpha - (x2 - x0) * sinAlpha + y0;
            if (y1 < 0 && y1 >= height) {
            	dest[x2*height + y2] = 0.0;
            	continue;
            }
            dest[x2*height + y2] = source[x1*height + y1];
        }
    }
}

void rotate(int height, int width, float *source, float *dest, float alpha, InterpolationType interpolation)
{
	if (interpolation == NONE)
		rotate_none(height, width, source, dest, alpha);
	else if (interpolation == BILINEAR)
		rotate_bilinear(height, width, source, dest, alpha);
	else {
		printf("FATAL ERROR: rotateAndCrop: unknown interpolation\n");
		abort();
	}
}

void flip(int height, int width, float *source, float *dest)
{
	float *pdest = dest + (height-1) * width;
	float *psource = source;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
		    pdest[j] = psource[j];
		}
		pdest -= width;
		psource += width;
	}
}

void crop(int height, int width, int height_new, int width_new, float * source, float *dest)
{
	int width_margin = (width - width_new) / 2;
	int height_margin = (height - height_new) / 2;

	//std::cout << "width_margin = " << width_margin << std::endl;
	//std::cout << "height_margin = " << height_margin << std::endl;

	for (int i = 0; i < height_new; ++i) {
		for (int j = 0; j < width_new; ++j) {
		    dest[i*width_new+j] = source[(i+height_margin)*width + (j+width_margin)];
		}
	}
}

void flipAndCrop(int height, int width, int height_new, int width_new, float *source, float *dest)
{
	int width_margin = (width - width_new) / 2;
	int height_margin = (height - height_new) / 2;

	for (int i = 0, ri = height_new-1; i < height_new; ++i, --ri) {
		for (int j = 0; j < width_new; ++j) {
		    dest[ri*width_new + j] = source[(i+height_margin)*width + (j+width_margin)];
		}
	}
}

void rotateAndCrop_bilinear(int height, int width, int height_new, int width_new, float *source, float *dest, float alpha)
{
	int width_margin = (width - width_new) / 2;
	int height_margin = (height - height_new) / 2;

    const float cosAlpha = cos(alpha);
    const float sinAlpha = sin(alpha);

    int x0 = width * 0.5;
    int y0 = height * 0.5;
    int x1, y1, x2, y2;

    for (x2 = 0; x2 < width_new; ++x2) {
        for (y2 = 0; y2 < height_new; ++y2) {
        	x1 = (x2 + width_margin - x0) * cosAlpha + (y2 + height_margin - y0) * sinAlpha + x0;
            if (x1 < 0 or x1 >= width) {
            	dest[x2*height_new + y2] = 0.0;
            	continue;
            }
        	y1 = (y2 + height_margin - y0) * cosAlpha - (x2 + width_margin - x0) * sinAlpha + y0;
            if (y1 < 0 && y1 >= height) {
            	dest[x2*height_new + y2] = 0.0;
            	continue;
            }
            dest[x2*height_new + y2] = source[x1*height + y1];
        }
    }
}

void rotateAndCrop(int height, int width, int height_new, int width_new, float *source, float *dest, float alpha, InterpolationType interpolation)
{
	if (interpolation == BILINEAR)
		rotateAndCrop_bilinear(height, width, height_new, width_new, source, dest, alpha);
	else {
		printf("FATAL ERROR: rotateAndCrop: unknown interpolation\n");
		abort();
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
    for (int i = 0; i < length; ++i, ++pa, ++pb) {
    	tmp = *pa - *pb;
        c += tmp * tmp;
    }
    return c;
}

void normalize(float *a, int length)
{
	float maxValue;
    for (int i = 0; i < length; ++i) {
        maxValue = fmax(maxValue, a[i]);
    }

    float maxValueInv;
    for (int i = 0; i < length; ++i) {
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

void fillZero(float *a, int length)
{
    for (int i = 0; i < length; ++i) {
    	a[i] = 0.0;
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
    #if PINK_USE_PYTHON
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
    #else
		std::cout << "=== WARNING === Pink must be compiled with python support to show images." << std::endl;
    #endif
}

void create_viewable_SOM(float* image, float* som, int som_dim, int image_dim)
{
	int total_image_dim = som_dim * image_dim;
    float *pimage = image;
    float *psom = som;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
            for (int k = 0; k < image_dim; ++k) {
                for (int l = 0; l < image_dim; ++l) {
        	        pimage[i*image_dim*som_dim*image_dim + k*image_dim*som_dim + j*image_dim + l] = *psom++;
            	}
            }
    	}
    }
}

void writeSOM(float* som, int som_dim, int image_dim, std::string const& filename)
{
	int total_image_dim = som_dim*image_dim;
	float *image = (float *)malloc(total_image_dim * total_image_dim * sizeof(float));
	create_viewable_SOM(image, som, som_dim, image_dim);
    writeImageToBinaryFile(image, total_image_dim, total_image_dim, filename);
    free(image);
}

void showSOM(float* som, int som_dim, int image_dim)
{
	int total_image_dim = som_dim*image_dim;
	float *image = (float *)malloc(total_image_dim * total_image_dim * sizeof(float));
	create_viewable_SOM(image, som, som_dim, image_dim);
    showImage(image, total_image_dim, total_image_dim);
    free(image);
}

void writeRotatedImages(float* images, int image_dim, int numberOfImages, std::string const& filename)
{
	int heigth = numberOfImages * image_dim;
	int width = image_dim;
	int image_size = image_dim * image_dim;
    float *image = (float *)malloc(heigth * width * sizeof(float));

    for (int i = 0; i < numberOfImages; ++i) {
        for (int j = 0; j < image_size; ++j) image[j + i*image_size] = images[j + i*image_size];
    }

    writeImageToBinaryFile(image, heigth, width, filename);
    free(image);
}

void showRotatedImages(float* images, int image_dim, int numberOfRotations)
{
	int heigth = 2 * numberOfRotations * image_dim;
	int width = image_dim;
	int image_size = image_dim * image_dim;
    float *image = (float *)malloc(heigth * width * sizeof(float));

    for (int i = 0; i < 2 * numberOfRotations; ++i) {
        for (int j = 0; j < image_size; ++j) image[j + i*image_size] = images[j + i*image_size];
    }

    showImage(image, heigth, width);
    free(image);
}

void showRotatedImagesSingle(float* images, int image_dim, int numberOfRotations)
{
	int image_size = image_dim * image_dim;
    for (int i = 0; i < 2 * numberOfRotations; ++i) {
        showImage(images + i*image_size, image_dim, image_dim);
    }
}
