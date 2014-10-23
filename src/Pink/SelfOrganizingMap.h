/**
 * @file   SelfOrganizingMap.h
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef SELFORGANIZINGMAP_H_
#define SELFORGANIZINGMAP_H_

#include <iostream>

//! Type for SOM layout.
enum Layout {QUADRATIC, HEXAGONAL};

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, Layout layout);

/**
 * @brief
 *
 * similarityMatrix
 */
void generateSimilarityMatrix(float *similarityMatrix, int som_dim, float* som, int image_dim, float* image);

/**
 * @brief
 *
 * similarityMatrix
 */
int findBestMatchingNeuron(float *similarityMatrix, int som_dim);

char* stringToUpper(char* s);

#endif /* SELFORGANIZINGMAP_H_ */
