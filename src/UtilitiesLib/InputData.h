/**
 * @file   InputData.h
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include "ImageProcessingLib/ImageProcessing.h"
#include "Version.h"
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

//! Type for storage of intermediate SOMs.
enum IntermediateStorageType {
    OFF,
    OVERWRITE,
    KEEP
};

//! Pretty printing of IntermediateStorageType.
std::ostream& operator << (std::ostream& os, IntermediateStorageType type);

#define DEFAULT_SIGMA     1.1
#define DEFAULT_DAMPING   0.2

struct InputData
{
    //! Default constructor.
    InputData();

    //! Constructor reading input data from arguments.
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
    int som_width;
    int som_height;
    int som_depth;
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
    int som_size;
    int neuron_size;
    int som_total_size;
    int numberOfRotationsAndFlip;
    Interpolation interpolation;
    ExecutionPath executionPath;
    IntermediateStorageType intermediate_storage;
    Function function;
    float sigma;
    float damping;
    int block_size_1;
    int maxUpdateDistance;
    int useMultipleGPUs;
    int usePBC;
    int dimensionality;
};

void stringToUpper(char* s);
