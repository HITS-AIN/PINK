/**
 * @file   ImageProcessing.h
 * @brief  Plain-C functions for image processing.
 * @date   Oct 7, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef IMAGEPROCESSING_H_
#define IMAGEPROCESSING_H_

#include <string>
#include <vector>

/**
 * @brief Interpolation type for images.
 */
enum Interpolation
{
	NEAREST_NEIGHBOR,  //!< Refuse values behind the comma.
	BILINEAR           //!< Interpolate value by distance to pixels.
};

//! Pretty printing of interpolation type.
std::ostream& operator << (std::ostream& os, Interpolation interpolation);

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
void rotate(int height, int width, float *source, float *dest, float alpha,
	Interpolation interpolation = BILINEAR);

/**
 * @brief Special rotation of 90 degrees clockwise.
 */
void rotate_90degrees(int height, int width, float *source, float *dest);

/**
 * @brief Plain-C function for image mirroring.
 */
void flip(int height, int width, float * source, float *dest);

/**
 * @brief Plain-C function for cropping an image.
 */
void crop(int height, int width, int height_new, int width_new, float * source, float *dest);

/**
 * @brief Plain-C function for flipping and cropping an image.
 *
 * The combined execution is more efficient.
 */
void flipAndCrop(int height, int width, int height_new, int width_new, float *source, float *dest);

/**
 * @brief Plain-C function for rotating and cropping an image.
 *
 * The combined execution is more efficient.
 */
void rotateAndCrop(int height, int width, int height_new, int width_new, float *source,
	float *dest, float alpha, Interpolation interpolation = BILINEAR);

/**
 * @brief Euclidean distance of two float arrays.
 *
 * Return sqrt(sum((a[i] - b[i])^2))
 */
float calculateEuclideanDistance(float *a, float *b, int length);

/**
 * @brief Same as @calculateEuclideanDistance but without square root to speed up.
 *
 * Return sum((a[i] - b[i])^2)
 */
float calculateEuclideanDistanceWithoutSquareRoot(float *a, float *b, int length);

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

//! For debugging: printing images on stdout.
void printImage(float *image, int height, int width);

void writeImagesToBinaryFile(std::vector<float> const& images, int numberOfImages, int numberOfChannels,
    int height, int width, std::string const& filename);

void readImagesFromBinaryFile(std::vector<float> &images, int &numberOfImages, int &numberOfChannels,
    int &height, int &width, std::string const& filename);

void showImage(float *image, int height, int width);

void convertSOMToImage(float *image, const float *som, int som_dim, int image_dim);

void convertImageToSOM(float *image, const float *som, int som_dim, int image_dim);

void writeSOM(const float *som, int numberOfChannels, int som_dim, int neuron_dim, std::string const& filename);

void readSOM(float *som, int &numberOfChannels, int &som_dim, int &neuron_dim, std::string const& filename);

void writeRotatedImages(float *images, int image_dim, int numberOfRotations, std::string const& filename);

void showRotatedImages(float *images, int image_dim, int numberOfRotations);

void showRotatedImagesSingle(float *images, int image_dim, int numberOfRotations);

#endif /* IMAGEPROCESSING_H_ */
