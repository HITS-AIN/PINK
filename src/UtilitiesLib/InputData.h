/**
 * @file   InputData.h
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef INPUTDATA_H_
#define INPUTDATA_H_

#include "ImageProcessingLib/ImageProcessing.h"
#include "Version.h"
#include <string>

//! Type for SOM layout.
enum Layout {
	QUADRATIC,
	QUADHEX,
	HEXAGONAL
};

//! Type for SOM initialization.
enum SOMInitialization {
	ZERO,
	RANDOM,
	RANDOM_WITH_PREFERRED_DIRECTION,
	FILEINIT
};

//! Type for execution path.
enum ExecutionPath {
	UNDEFINED,
	TRAIN,
	MAP
};

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, Layout layout);

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, SOMInitialization init);

struct InputData
{
	//! Read input data from arguments.
	InputData(int argc, char **argv);

	//! Print program header.
	void print_header() const;

	//! Print input data.
	void print_parameters() const;

	//! Print usage output for input arguments.
	void print_usage() const;

	std::string imagesFilename;
	std::string resultFilename;
	std::string somFilename;
    std::string initSomFilename;

	bool verbose;
	int som_dim;
	int neuron_dim;
	Layout layout;
	int seed;
	int numberOfRotations;
	int numberOfThreads;
	SOMInitialization init;
	int numIter;
	float progressFactor;
	bool useFlip;
	bool useCuda;
	int numberOfImages;
	int numberOfChannels;
	int image_dim;
	int image_size;
	int image_size_using_flip;
    int som_size;
    int neuron_size;
    int som_total_size;
	int numberOfRotationsAndFlip;
	int algo;
	Interpolation interpolation;
	ExecutionPath executionPath;
};

void stringToUpper(char* s);

#endif /* INPUTDATA_H_ */
