/**
 * @file   SelfOrganizingMap.cpp
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include <stdlib.h>
#include <ctype.h>

std::ostream& operator << (std::ostream& os, Layout layout)
{
	if (layout == QUADRATIC) os << "quadratic";
	else if (layout == HEXAGONAL) os << "hexagonal";
	else os << "undefined";
	return os;
}

void generateSimilarityMatrix(float *similarityMatrix, int som_dim, float* som, int image_dim, float* image)
{
	int som_size = som_dim * som_dim;
	int image_size = image_dim * image_dim;

    for (int i = 0; i < som_size; ++i) {
    	similarityMatrix[i] = calculateEuclideanSimilarity(som + i*image_size, image, image_size);
    }

    float maxSimilarity = 0.0;
    int maxSimilarityIndex = 0;

    for (int i = 0; i < som_size; ++i) {
    	if (similarityMatrix[i] > maxSimilarity) {
    		maxSimilarity = similarityMatrix[i];
    		maxSimilarityIndex = i;
    	}
    }
}

int findBestMatchingNeuron(float *similarityMatrix, int som_dim)
{
	int som_size = som_dim * som_dim;
    float maxSimilarity = 0.0;
    int maxSimilarityIndex = 0;

    for (int i = 0; i < som_size; ++i) {
    	if (similarityMatrix[i] > maxSimilarity) {
    		maxSimilarity = similarityMatrix[i];
    		maxSimilarityIndex = i;
    	}
    }

    return maxSimilarityIndex;
}

char* stringToUpper(char* s)
{
	for (; *s != '\0'; ++s)
	{
		*s = tolower(*s);
	}
	return s;
}
