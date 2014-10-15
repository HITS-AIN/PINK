/*
 * ImageProcessing.c
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessing.h"
#include <math.h>

// x2 = cos(alpha) * (x1 - x0) - sin(alpha) * (y1 - y0) + x0
// y2 = sin(alpha) * (x1 - x0) + cos(alpha) * (y1 - y0) + y0

// (x0,y0) is the center of rotation

void rotate(int height, int width, float *source, float *dest, float alpha)
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
