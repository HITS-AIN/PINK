/**
 * @file   ImageProcessing.c
 * @brief  Plain-C functions for image processing.
 * @date   Oct 7, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessing.h"
#include <math.h>
#include <random>
#include <stdlib.h>

void rotate_none(int height, int width, float *source, float *dest, float alpha)
{
	int x0, x1, x2, y0, y1, y2;
    const float cosAlpha = cos(alpha);
    const float sinAlpha = sin(alpha);

    x0 = width / 2;
    y0 = height / 2;

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
	float x0, x1, x2, y0, y1, y2;
    const float cosAlpha = cos(alpha);
    const float sinAlpha = sin(alpha);

    x0 = width * 0.5;
    y0 = height * 0.5;

    for (x1 = 0; x1 < width; ++x1) {
        for (y1 = 0; y1 < height; ++y1) {
        	x2 = (x1 - x0) * cosAlpha - (y1 - y0) * sinAlpha + x0;
        	y2 = (x1 - x0) * sinAlpha + (y1 - y0) * cosAlpha + y0;
            //if (x2 > -1 && x2 < width && y2 > -1 && y2 < height) dest[x2*height + y2] = source[x1*height + y1];
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

float calculateEuclideanSimilarity(float *a, float *b, int length)
{
	int i;
	float c = 0.0;
    for (i = 0; i < length; ++i) {
        c += pow((a[i] - b[i]),2);
    }
    return sqrt(c);
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
