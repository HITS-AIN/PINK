/**
 * @file   SelfOrganizingMap.cpp
 * @brief  Plain-C functions for self organizing map.
 * @date   Oct 23, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include <ctype.h>
#include <float.h>
#include <stdlib.h>
#include <cmath>

std::ostream& operator << (std::ostream& os, Layout layout)
{
	if (layout == QUADRATIC) os << "quadratic";
	else if (layout == HEXAGONAL) os << "hexagonal";
	else os << "undefined";
	return os;
}

std::ostream& operator << (std::ostream& os, Point p)
{
	return os << "(" << p.x << "," << p.y << ")";
}

void generateRotatedImages(float *rotatedImages, float *image, int numberOfRotations, int image_dim)
{
	int image_size = image_dim * image_dim;
	float angleStepRadians = 360.0 * M_PI / (numberOfRotations * 180.0);

	// Copy original image on first position
	for (int i = 0; i < image_size; ++i) {
		rotatedImages[i] = image[i];
	}

	// Rotate unfliped image
	for (int i = 1; i < numberOfRotations; ++i)	{
		rotate(image_dim, image_dim, image, rotatedImages + i*image_size, i * angleStepRadians);
	}

	// Flip image
	float *flippedImage = rotatedImages + numberOfRotations * image_size;
	flip(image_dim, image_dim, image, flippedImage);

	// Rotate fliped image
	for (int i = 1; i < numberOfRotations; ++i)	{
		rotate(image_dim, image_dim, flippedImage, flippedImage + i*image_size, i * angleStepRadians);
	}
}

void generateSimilarityMatrix(float *similarityMatrix, int *bestRotationMatrix, int som_dim, float* som,
	int image_dim, int numberOfRotations, float* image)
{
	int som_size = som_dim * som_dim;
	int image_size = image_dim * image_dim;

	float simTmp;
	float* psim = similarityMatrix;
	int* prot = bestRotationMatrix;

    for (int i = 0; i < som_size; ++i) similarityMatrix[i] = FLT_MAX;

    for (int i = 0; i < som_size; ++i, ++psim, ++prot) {
        for (int j = 0; j < numberOfRotations; ++j) {
    	    simTmp = calculateEuclideanSimilarity(som + i*image_size, image + j*image_size, image_size);
    	    if (simTmp < *psim) {
    	    	*psim = simTmp;
                *prot = j;
    	    }
        }
    }
}

Point findBestMatchingNeuron(float *similarityMatrix, int som_dim)
{
	int som_size = som_dim * som_dim;
    float maxSimilarity = 0.0;
    Point bestMatch;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
			if (similarityMatrix[i*som_dim+j] > maxSimilarity) {
				maxSimilarity = similarityMatrix[i*som_dim+j];
				bestMatch.x = i;
				bestMatch.y = j;
			}
		}
    }

    return bestMatch;
}

void updateNeurons(int som_dim, float* som, int image_dim, float* image, Point const& bestMatch, int *bestRotationMatrix)
{
	float factor;
	int image_size = image_dim * image_dim;
	float *current_neuron = som;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
        	factor = gaussian(distance(bestMatch,Point(i,j)), UPDATE_NEURONS_SIGMA) * UPDATE_NEURONS_DAMPING;
        	updateSingleNeuron(current_neuron, image + bestRotationMatrix[i*som_dim+j], image_size, factor);
        	current_neuron += image_size;
    	}
    }
}

void updateSingleNeuron(float* neuron, float* image, int image_size, float factor)
{
    for (int i = 0; i < image_size; ++i) {
    	neuron[i] -= (neuron[i] - image[i]) * factor;
    }
}

void showSOM(float* som, int som_dim, int image_dim)
{
    PINK::Image<float> image(som_dim*image_dim,som_dim*image_dim);
    float *pixel = image.getPointerOfFirstPixel();
    float *psom = som;

    for (int i = 0; i < som_dim; ++i) {
        for (int j = 0; j < som_dim; ++j) {
            for (int k = 0; k < image_dim; ++k) {
                for (int l = 0; l < image_dim; ++l) {
        	        pixel[i*image_dim*som_dim*image_dim + k*image_dim*som_dim + j*image_dim + l] = *psom++;
            	}
            }
    	}
    }

    image.show();
}

void showRotatedImages(float* images, int image_dim, int numberOfRotations)
{
	int image_size = image_dim * image_dim;
    PINK::Image<float> image(3*image_dim,image_dim);
    float *pixel = image.getPointerOfFirstPixel();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < image_size; ++j) pixel[j + i*image_size] = images[j + i*image_size];
    }
    image.show();
}

float distance(Point pos1, Point pos2)
{
    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

char* stringToUpper(char* s)
{
	for (; *s != '\0'; ++s)
	{
		*s = tolower(*s);
	}
	return s;
}

// 2.0 / ( math.sqrt(3.0 * sigma) * math.pow(math.pi, 0.25)) * (1- x**2.0 / sigma**2.0) * math.exp(-x**2.0/(2.0 * sigma**2))
float mexicanHat(float x, float sigma)
{
	float x2 = x * x;
	float sigma2 = sigma * sigma;
    return 2.0 / (sqrt(3.0 * sigma) * pow(M_PI, 0.25)) * (1.0 - x2/sigma2) * exp(-x2 / (2.0 * sigma2));
}

// 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-1.0/2.0 * (x / sigma)**2 );
float gaussian(float x, float sigma)
{
    return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * pow((x/sigma),2));
}
