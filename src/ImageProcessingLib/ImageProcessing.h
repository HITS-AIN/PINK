/*
 * ImageProcessing.h
 *
 *  Created on: Oct 7, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#ifndef IMAGEPROCESSING_H_
#define IMAGEPROCESSING_H_

//! Plain-C function for image rotation
//! angle in radians
void rotate(int height, int width, float *source, float *dest, float angle);

float calculateEuclideanSimilarity(float *a, float *b);

#endif /* IMAGEPROCESSING_H_ */
