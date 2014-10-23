/**
 * @file   ImageProcessing.h
 * @brief  Plain-C functions for image processing.
 * @date   Oct 7, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef IMAGEPROCESSING_H_
#define IMAGEPROCESSING_H_

enum InterpolationType { NONE, BILINEAR };

/**
 * @brief Plain-C function for image rotation.
 *
 * Angle (alpha) in radians
 *
 * Old position: (x1,y1)
 * New position: (x2,y2)
 * Center of rotation: (x0,y0)
 *
 * x2 = cos(alpha) * (x1 - x0) - sin(alpha) * (y1 - y0) + x0
 * y2 = sin(alpha) * (x1 - x0) + cos(alpha) * (y1 - y0) + y0
 */
void rotate(int height, int width, float *source, float *dest, float alpha, InterpolationType interpolation = BILINEAR);

/**
 * @brief Similarity of to float arrays using euclidean norm.
 *
 * Return sqrt((a[i] - b[i])^2)
 */
float calculateEuclideanSimilarity(float *a, float *b, int length);

/**
 * @brief Normalize image values.
 *
 * Divide by max value.
 */
void normalize(float *a, int length);

/**
 * @brief Arithmetic mean value.
 *
 * Sum of the elements divided by the number of elements.
 */
float mean(float *a, int length);

/**
 * @brief Standard deviation.
 *
 * Returns standard deviation.
 */
float stdDeviation(float *a, int length);

/**
 * @brief Normalize image values.
 *
 * Set values smaller than safety * stdDeviation to zero.
 * stdDeviation = sqrt(mean(abs(x - x.mean())**2)).
 */
void zeroValuesSmallerThanStdDeviation(float *a, int length, float safety);

/**
 * @brief Fill array with random numbers
 */
void fillRandom(float *a, int length, int seed = 0);

#endif /* IMAGEPROCESSING_H_ */
