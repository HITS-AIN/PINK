/**
 * @file   InputData.h
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "IntermediateStorageType.h"
#include "SOMInitializationType.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/DistributionFunctor.h"
#include "UtilitiesLib/ExecutionPath.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/Interpolation.h"
#include "Version.h"

namespace pink {

struct InputData
{
    /// Default constructor
    InputData();

    /// Constructor reading input data from arguments
    InputData(int argc, char **argv);

    /// Print program header
    void print_header() const;

    /// Print input data
    void print_parameters() const;

    /// Print usage output for input arguments
    void print_usage() const;

    /// Return the distribution function
    std::function<float(float)> get_distribution_function() const;

    std::string data_filename;
    std::string result_filename;
    std::string som_filename;
    std::string rot_flip_filename;

    bool verbose;
    uint32_t som_width;
    uint32_t som_height;
    uint32_t som_depth;
    uint32_t neuron_dim;
    uint32_t euclidean_distance_dim;
    Layout layout;
    int seed;
    int number_of_rotations;
    int number_of_threads;
    SOMInitialization init;
    uint32_t number_of_iterations;
    int max_number_of_progress_prints;
    bool use_flip;
    bool use_gpu;
    uint32_t number_of_data_entries;
    Layout data_layout;
    std::vector<uint32_t> data_dimension;
    uint32_t som_size;
    uint32_t neuron_size;
    uint32_t som_total_size;
    int number_of_spatial_transformations;
    Interpolation interpolation;
    ExecutionPath executionPath;
    IntermediateStorageType intermediate_storage;
    DistributionFunction distribution_function;
    float sigma;
    float damping;
    int block_size_1;
    float max_update_distance;
    int usePBC;
    int dimensionality;
    bool write_rot_flip;
    DataType euclidean_distance_type;
    bool shuffle_data_input;
};

void stringToUpper(char* s);

} // namespace pink
