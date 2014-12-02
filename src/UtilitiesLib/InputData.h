/**
 * @file   InputData.h
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef INPUTDATA_H_
#define INPUTDATA_H_

#include "ImageProcessingLib/ImageProcessing.h"
#include "UtilitiesLib/DistributionFunctions.h"
#include "Version.h"
#include <memory>
#include <string>

//! Type for SOM layout.
enum Layout {
	QUADRATIC,
	QUADHEX,
	HEXAGONAL
};

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, Layout layout);

//! Type for distribution function for SOM update.
enum Function {
    GAUSSIAN,
    MEXICANHAT
};

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, Function function);

//! Type for SOM initialization.
enum SOMInitialization {
	ZERO,
	RANDOM,
	RANDOM_WITH_PREFERRED_DIRECTION,
	FILEINIT
};

//! Pretty printing of SOM layout type.
std::ostream& operator << (std::ostream& os, SOMInitialization init);

//! Type for execution path.
enum ExecutionPath {
	UNDEFINED,
	TRAIN,
	MAP
};

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
	bool intermediate_storage;
	Function function;
	float sigma;
	float damping;

	//std::shared_ptr<DistributionFunctionBase> ptrDistributionFunctor;
    //std::shared_ptr<DistanceFunctor> ptrDistanceFunctor;
};

void stringToUpper(char* s);

#endif /* INPUTDATA_H_ */
