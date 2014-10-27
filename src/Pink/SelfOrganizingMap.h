/**
 * @file   SelfOrganizingMap.h
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef SELFORGANIZINGMAP_H_
#define SELFORGANIZINGMAP_H_

#include <iostream>

#define UPDATE_NEURONS_SIGMA     1.1
#define UPDATE_NEURONS_DAMPING   0.2

struct Point
{
	Point(int x = 0, int y = 0) : x(x), y(y) {}

	int x;
	int y;
};

//! Pretty printing of Point.
std::ostream& operator << (std::ostream& os, Point point);

//! Type for SOM layout.
enum Layout {QUADRATIC, HEXAGONAL};

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, Layout layout);

void generateRotatedImages(float *rotatedImages, float *image, int numberOfRotations, int image_dim);

void generateSimilarityMatrix(float *similarityMatrix, int *bestRotationMatrix, int som_dim, float* som,
	int image_dim, int numberOfRotations, float* image);

/**
 * Returns the position of the best matching neuron.
 */
Point findBestMatchingNeuron(float *similarityMatrix, int som_dim);

/**
 * @brief Updating SOM
 */
void updateNeurons(int som_dim, float* som, int image_dim, float* image, Point const& bestMatch, int *bestRotationMatrix);

void updateSingleNeuron(float* neuron, float* image, int image_size, float factor);

void showSOM(float* som, int som_dim, int image_dim);

void showRotatedImages(float* images, int image_dim, int numberOfRotations);

float distance(Point pos1, Point pos2);

char* stringToUpper(char* s);

float mexicanHat(float x, float sigma);

float gaussian(float x, float sigma);

#endif /* SELFORGANIZINGMAP_H_ */
