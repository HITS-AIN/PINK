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
#include "UtilitiesLib/EuclideanDistanceShape.h"
#include "UtilitiesLib/ExecutionPath.h"
#include "UtilitiesLib/Interpolation.h"
#include "UtilitiesLib/Layout.h"
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

    std::string m_data_filename;
    std::string m_result_filename;
    std::string m_som_filename;
    std::string m_rot_flip_filename;

    bool m_verbose;
    uint32_t m_som_width;
    uint32_t m_som_height;
    uint32_t m_som_depth;
    uint32_t m_neuron_dim;
    uint32_t m_euclidean_distance_dim;
    Layout m_layout;
    uint32_t m_seed;
    uint32_t m_number_of_rotations;
    int m_number_of_threads;
    SOMInitialization m_init;
    uint32_t m_number_of_iterations;
    int m_max_number_of_progress_prints;
    bool m_use_flip;
    bool m_use_gpu;
    uint32_t m_number_of_data_entries;
    Layout m_data_layout;
    std::vector<uint32_t> m_data_dimension;
    std::vector<uint32_t> m_neuron_dimension;
    uint32_t m_som_size;
    uint32_t m_neuron_size;
    uint32_t m_som_total_size;
    uint32_t m_number_of_spatial_transformations;
    Interpolation m_interpolation;
    ExecutionPath m_executionPath;
    IntermediateStorageType m_intermediate_storage;
    DistributionFunction m_distribution_function;
    float m_sigma;
    float m_damping;
    uint32_t m_block_size_1;
    float m_max_update_distance;
    int m_use_pbc;
    int m_dimensionality;
    bool m_write_rot_flip;
    DataType m_euclidean_distance_type;
    bool m_shuffle_data_input;
    EuclideanDistanceShape m_euclidean_distance_shape;
};

} // namespace pink
