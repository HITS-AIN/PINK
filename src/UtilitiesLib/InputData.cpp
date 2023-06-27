/**
 * @file   InputData.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>

#include "InputData.h"
#include "pink_exception.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "UtilitiesLib/get_file_header.h"

namespace {

uint32_t str_to_uint32_t(std::string const& str)
{
    auto i = std::stoi(str);
    if (i < 0) throw std::runtime_error("str_to_uint32_t: integer must be positive");
    return static_cast<uint32_t>(i);
}

std::string str_to_upper(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c) {
            return std::toupper(c);
        }
    );
    return str;
}

} // end anonymous namespace

namespace pink {

InputData::InputData()
 : m_verbose(false),
   m_som_width(10),
   m_som_height(10),
   m_som_depth(1),
   m_neuron_dim(0),
   m_euclidean_distance_dim(0),
   m_layout(Layout::CARTESIAN),
   m_seed(1234),
   m_number_of_rotations(360),
   m_number_of_threads(-1),
   m_init(SOMInitialization::ZERO),
   m_number_of_iterations(1),
   m_max_number_of_progress_prints(10),
   m_use_flip(true),
   m_use_gpu(true),
   m_number_of_data_entries(0),
   m_data_layout(Layout::CARTESIAN),
   m_som_size(0),
   m_neuron_size(0),
   m_som_total_size(0),
   m_number_of_spatial_transformations(0),
   m_interpolation(Interpolation::BILINEAR),
   m_executionPath(ExecutionPath::UNDEFINED),
   m_intermediate_storage(IntermediateStorageType::OFF),
   m_distribution_function(DistributionFunction::GAUSSIAN),
   m_sigma(1.1f),
   m_damping(0.2f),
   m_block_size_1(256),
   m_max_update_distance(-1.0),
   m_use_pbc(false),
   m_dimensionality(1),
   m_write_rot_flip(false),
   m_euclidean_distance_type(DataType::UINT8),
   m_shuffle_data_input(true),
   m_euclidean_distance_shape(EuclideanDistanceShape::QUADRATIC)
{}

InputData::InputData(int argc, char **argv)
 : InputData()
{
    static struct option long_options[] = {
        {"neuron-dimension",             1, nullptr, 'd'},
        {"euclidean-distance-dimension", 1, nullptr, 'e'},
        {"layout",                       1, nullptr, 'l'},
        {"seed",                         1, nullptr, 's'},
        {"numrot",                       1, nullptr, 'n'},
        {"numthreads",                   1, nullptr, 't'},
        {"init",                         1, nullptr, 'x'},
        {"progress",                     1, nullptr, 'p'},
        {"version",                      0, nullptr, 'v'},
        {"help",                         0, nullptr, 'h'},
        {"dist-func",                    1, nullptr, 'f'},
        {"som-width",                    1, nullptr, 0},
        {"num-iter",                     1, nullptr, 1},
        {"flip-off",                     0, nullptr, 2},
        {"cuda-off",                     0, nullptr, 3},
        {"verbose",                      0, nullptr, 4},
        {"interpolation",                1, nullptr, 5},
        {"train",                        1, nullptr, 6},
        {"map",                          1, nullptr, 7},
        {"inter-store",                  1, nullptr, 8},
        {"b1",                           1, nullptr, 9},
        {"max-update-distance",          1, nullptr, 10},
        {"som-height",                   1, nullptr, 12},
        {"som-depth",                    1, nullptr, 13},
        {"pbc",                          0, nullptr, 14},
        {"store-rot-flip",               1, nullptr, 15},
        {"euclidean-distance-type",      1, nullptr, 16},
        {"input-shuffle-off",            0, nullptr, 17},
        {"euclidean-distance-shape" ,    1, nullptr, 18},
        {nullptr,                        0, nullptr, 0}
    };

    int c = 0;
    int option_index = 0;
    char *end_char;

    while ((c = getopt_long(argc, argv, "vd:l:s:n:t:x:p:a:hf:", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'd':
            {
                m_neuron_dim = str_to_uint32_t(optarg);
                break;
            }
            case 'e':
            {
                m_euclidean_distance_dim = str_to_uint32_t(optarg);
                break;
            }
            case 0:
            {
                m_som_width = str_to_uint32_t(optarg);
                break;
            }
            case 12:
            {
                m_som_height = str_to_uint32_t(optarg);
                break;
            }
            case 13:
            {
                m_som_depth = str_to_uint32_t(optarg);
                break;
            }
            case 1:
            {
                m_number_of_iterations = str_to_uint32_t(optarg);
                if (m_number_of_iterations == 0)
                    throw pink::exception("Number of iterations must be larger than 0");
                break;
            }
            case 'l':
            {
                auto str = str_to_upper(optarg);
                if (str == "CARTESIAN") {
                    m_layout = Layout::CARTESIAN;
                }
                else if (str == "HEXAGONAL") {
                    m_layout = Layout::HEXAGONAL;
                }
                else {
                    throw pink::exception("Unknown layout option " + str);
                }
                break;
            }
            case 's':
            {
                m_seed = str_to_uint32_t(optarg);
                break;
            }
            case 'p':
            {
                m_max_number_of_progress_prints = std::atoi(optarg);
                break;
            }
            case 'n':
            {
                m_number_of_rotations = str_to_uint32_t(optarg);
                if (m_number_of_rotations == 0 or (m_number_of_rotations != 1 and m_number_of_rotations % 4))
                    throw pink::exception("Number of rotations must be 1 or a multiple of 4");
                break;
            }
            case 't':
            {
                m_number_of_threads = atoi(optarg);
                break;
            }
            case 'x':
            {
                auto str = str_to_upper(optarg);
                if (str == "ZERO") {
                    m_init = SOMInitialization::ZERO;
                }
                else if (str == "RANDOM") {
                    m_init = SOMInitialization::RANDOM;
                }
                else if (str == "RANDOM_WITH_PREFERRED_DIRECTION") {
                    m_init = SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION;
                }
                else {
                    m_init = SOMInitialization::FILEINIT;
                    m_som_filename = optarg;
                }
                break;
            }
            case 2:
            {
                m_use_flip = false;
                break;
            }
            case 3:
            {
                m_use_gpu = false;
                break;
            }
            case 4:
            {
                m_verbose = true;
                break;
            }
            case 5:
            {
                auto str = str_to_upper(optarg);
                if (str == "NEAREST_NEIGHBOR") {
                    m_interpolation = Interpolation::NEAREST_NEIGHBOR;
                }
                else if (str == "BILINEAR") {
                    m_interpolation = Interpolation::BILINEAR;
                }
                else {
                    throw pink::exception("Unknown interpolation option " + str);
                }
                break;
            }
            case 6:
            {
                m_executionPath = ExecutionPath::TRAIN;
                int index = optind - 1;
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --train option.");
                }
                m_data_filename = argv[index++];
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --train option.");
                }
                m_result_filename = argv[index++];
                optind = index - 1;
                break;
            }
            case 7:
            {
                m_executionPath = ExecutionPath::MAP;
                int index = optind - 1;
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --map option.");
                }
                m_data_filename = argv[index++];
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --map option.");
                }
                m_result_filename = argv[index++];
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --map option.");
                }
                m_som_filename = argv[index++];
                optind = index - 1;
                break;
            }
            case 8:
            {
                auto str = str_to_upper(optarg);
                if (str == "OFF") {
                    m_intermediate_storage = IntermediateStorageType::OFF;
                }
                else if (str == "OVERWRITE") {
                    m_intermediate_storage = IntermediateStorageType::OVERWRITE;
                }
                else if (str == "KEEP") {
                    m_intermediate_storage = IntermediateStorageType::KEEP;
                }
                else {
                    throw pink::exception("Unknown intermediate storage option " + str);
                }
                break;
            }
            case 9:
            {
                m_block_size_1 = str_to_uint32_t(optarg);
                break;
            }
            case 10:
            {
                m_max_update_distance = std::strtof(optarg, &end_char);
                if (m_max_update_distance <= 0.0f) {
                    print_usage();
                    throw pink::exception("max-update-distance must be positive.");
                }
                break;
            }
            case 14:
            {
                m_use_pbc = true;
                break;
            }
            case 15:
            {
                m_write_rot_flip = true;
                m_rot_flip_filename = optarg;
                break;
            }
            case 16:
            {
                auto str = str_to_upper(optarg);
                if (str == "FLOAT") {
                    m_euclidean_distance_type = DataType::FLOAT;
                }
                else if (str == "UINT16") {
                    m_euclidean_distance_type = DataType::UINT16;
                }
                else if (str == "UINT8") {
                    m_euclidean_distance_type = DataType::UINT8;
                }
                else {
                    throw pink::exception("Unknown intermediate storage option " + str);
                }
                break;
            }
            case 17:
            {
                m_shuffle_data_input = false;
                break;
            }
            case 18:
            {
                auto str = str_to_upper(optarg);
                if (str == "QUADRATIC") {
                    m_euclidean_distance_shape = EuclideanDistanceShape::QUADRATIC;
                }
                else if (str == "CIRCULAR") {
                    m_euclidean_distance_shape = EuclideanDistanceShape::CIRCULAR;
                }
                else {
                    throw pink::exception("Unknown euclidean distance shape " + str);
                }
                break;
            }
            case 'v':
            {
                std::cout << "Pink version " << PROJECT_VERSION << std::endl;
                std::cout << "Git revision " << GIT_REVISION << std::endl;
                exit(0);
            }
            case 'h':
            {
                print_usage();
                exit(0);
            }
            case 'f':
            {
                auto str = str_to_upper(optarg);
                if (str == "GAUSSIAN") {
                    m_distribution_function = DistributionFunction::GAUSSIAN;
                }
                else if (str == "UNITYGAUSSIAN") {
                    m_distribution_function = DistributionFunction::UNITYGAUSSIAN;
                }
                else if (str == "MEXICANHAT") {
                    m_distribution_function = DistributionFunction::MEXICANHAT;
                }
                else {
                    throw pink::exception("Unknown intermediate storage option " + str);
                }
                int index = optind;
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --dist-func option.");
                }
                m_sigma = std::strtof(argv[index++], &end_char);
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --dist-func option.");
                }
                m_damping = std::strtof(argv[index++], &end_char);
                optind = index;
                break;
            }
            case '?':
            {
                printf ("Unknown option %o\n", c);
                print_usage();
                exit(EXIT_FAILURE);
            }
            default:
            {
                printf ("Unknown option %o\n", c);
                print_usage();
                exit(EXIT_FAILURE);
            }
        }
    }

    if (m_executionPath == ExecutionPath::MAP) {
        m_init = SOMInitialization::FILEINIT;
    } else if (m_executionPath == ExecutionPath::UNDEFINED) {
        print_usage();
        throw pink::exception("Unknown execution path.");
    }

    if (m_layout == Layout::HEXAGONAL) {
        if (m_use_pbc) throw pink::exception("Periodic boundary conditions are not supported for hexagonal layout.");
        if ((m_som_width - 1) % 2) throw pink::exception("For hexagonal layout only odd dimension supported.");
        if (m_som_width != m_som_height) {
            throw pink::exception("For hexagonal layout som-width must be equal to som-height.");
        }
        if (m_som_depth != 1) throw pink::exception("For hexagonal layout som-depth must be equal to 1.");
        m_som_size = HexagonalLayout({m_som_width, m_som_height}).size();
    }
    else m_som_size = m_som_width * m_som_height * m_som_depth;

    if (m_som_width < 2) throw pink::exception("som-width must be > 1.");
    if (m_som_height < 1) throw pink::exception("som-height must be > 0.");
    if (m_som_depth < 1) throw pink::exception("som-depth must be > 0.");
    if (m_som_height > 1) ++m_dimensionality;
    if (m_som_depth > 1) ++m_dimensionality;

    std::ifstream ifs(m_data_filename);
    if (!ifs) throw std::runtime_error("Error opening " + m_data_filename);

    // Skip header
    get_file_header(ifs);

    int file_version, file_type, data_type;
    // Ignore first three entries
    ifs.read(reinterpret_cast<char*>(&file_version), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&file_type), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&data_type), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&m_number_of_data_entries), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&m_data_layout), sizeof(int));

    int data_dimensionality;
    ifs.read(reinterpret_cast<char*>(&data_dimensionality), sizeof(int));
    m_data_dimension.resize(static_cast<size_t>(data_dimensionality));

    for (size_t i = 0; i < static_cast<size_t>(data_dimensionality); ++i) {
        ifs.read(reinterpret_cast<char*>(&m_data_dimension[i]), sizeof(int));
    }

    if (m_neuron_dim == 0) {
        m_neuron_dim = m_data_dimension[1];
        if (m_number_of_rotations != 1)
            m_neuron_dim = static_cast<uint32_t>(2 * m_data_dimension[1] / std::sqrt(2.0) + 1);
    }
    assert(m_neuron_dim != 0);

    m_neuron_dimension = m_data_dimension;
    if (m_neuron_dimension.size() == 2) {
        m_neuron_dimension[0] = m_neuron_dim;
        m_neuron_dimension[1] = m_neuron_dim;
    }

    if (m_neuron_dimension.size() == 3) {
        m_neuron_dimension[1] = m_neuron_dim;
        m_neuron_dimension[2] = m_neuron_dim;
    }

    if (m_euclidean_distance_dim == 0) {
        m_euclidean_distance_dim = m_data_dimension[1];
        if (m_number_of_rotations != 1 and m_euclidean_distance_shape == EuclideanDistanceShape::QUADRATIC) {
            m_euclidean_distance_dim = static_cast<uint32_t>(m_euclidean_distance_dim * std::sqrt(2.0) / 2);
        }
    }
    assert(m_euclidean_distance_dim != 0);

    m_neuron_size = m_neuron_dim * m_neuron_dim;
    m_som_total_size = m_som_size * m_neuron_size;
    m_number_of_spatial_transformations = m_use_flip ? 2 * m_number_of_rotations : m_number_of_rotations;

    if (m_number_of_threads == -1) m_number_of_threads = omp_get_max_threads();
    omp_set_num_threads(m_number_of_threads);

    print_header();
    print_parameters();

    if (file_version != 2) throw pink::exception("Please use file format version 2 as data input.");
    if (file_type != 0) throw pink::exception("Please use file type 0 as data input.");
    if (data_type != 0) throw pink::exception("Only data_type = 0 (float, single precision) is supported.");
    if (m_number_of_data_entries == 0) throw pink::exception("Number of data entries must be larger than 0.");
    if (m_euclidean_distance_dim > m_neuron_dim)
        throw pink::exception("euclidean distance dimension must be equal or smaller than neuron dimension.");
    if (m_use_pbc) throw pink::exception("Periodic boundary conditions are not supported in version 2.");
}

void InputData::print_header() const
{
    std::cout << "\n"
                 "  *************************************************************************\n"
                 "  *                                                                       *\n"
                 "  *                    PPPPP    II   NN    NN   KK  KK                    *\n"
                 "  *                    PP  PP   II   NNN   NN   KK KK                     *\n"
                 "  *                    PPPPP    II   NN NN NN   KKKK                      *\n"
                 "  *                    PP       II   NN   NNN   KK KK                     *\n"
                 "  *                    PP       II   NN    NN   KK  KK                    *\n"
                 "  *                                                                       *\n"
                 "  *       Parallelized rotation and flipping INvariant Kohonen maps       *\n"
                 "  *                                                                       *\n"
                 "  *                         Version " << PROJECT_VERSION << "                                   *\n"
                 "  *                         Git revision: " << GIT_REVISION << "                         *\n"
                 "  *                                                                       *\n"
                 "  *       Bernd Doser <bernd.doser@h-its.org>                             *\n"
                 "  *       Kai Polsterer <kai.polsterer@h-its.org>                         *\n"
                 "  *                                                                       *\n"
                 "  *       Distributed under the GNU GPLv3 License.                        *\n"
                 "  *       See accompanying file LICENSE or                                *\n"
                 "  *       copy at http://www.gnu.org/licenses/gpl-3.0.html.               *\n"
                 "  *                                                                       *\n"
                 "  *************************************************************************\n"
              << std::endl;
}

void InputData::print_parameters() const
{
    std::cout << "  Data file = " << m_data_filename << "\n"
              << "  Result file = " << m_result_filename << "\n";

    if (m_executionPath == ExecutionPath::MAP)
        std::cout << "  SOM file = " << m_som_filename << "\n";

    std::cout << "  Number of data entries = " << m_number_of_data_entries << "\n"
              << "  Data dimension = " << m_data_dimension[0];

    for (size_t i = 1; i < m_data_dimension.size(); ++i) std::cout << " x " << m_data_dimension[i];
    std::cout << std::endl;

    std::cout << "  SOM dimension (width x height x depth) = "
              << m_som_width << "x" << m_som_height << "x" << m_som_depth << "\n"
              << "  SOM size = " << m_som_size << "\n"
              << "  Number of iterations = " << m_number_of_iterations << "\n"
              << "  Neuron dimension = " << m_neuron_dim << "x" << m_neuron_dim << "\n"
              << "  Euclidean distance dimension = " << m_euclidean_distance_dim << "\n"
              << "  Data type for euclidean distance calculation = " << m_euclidean_distance_type << "\n"
              << "  Shape of euclidean distance region = " << m_euclidean_distance_shape << "\n"
              << "  Maximal number of progress information prints = " << m_max_number_of_progress_prints << "\n"
              << "  Intermediate storage of SOM = " << m_intermediate_storage << "\n"
              << "  Layout = " << m_layout << "\n"
              << "  Initialization type = " << m_init;

    if (m_init == SOMInitialization::FILEINIT) std::cout << "\n  SOM initialization file = " << m_som_filename;

    std::cout << "\n"
              << "  Interpolation type = " << m_interpolation << "\n"
              << "  Seed = " << m_seed << "\n"
              << "  Number of rotations = " << m_number_of_rotations << "\n"
              << "  Use mirrored image = " << m_use_flip << "\n"
              << "  Number of CPU threads = " << m_number_of_threads << "\n"
              << "  Use CUDA = " << m_use_gpu << "\n";

    if (m_executionPath == ExecutionPath::TRAIN) {
        std::cout << "  Distribution function for SOM update = " << m_distribution_function << "\n"
                  << "  Sigma = " << m_sigma << "\n"
                  << "  Damping factor = " << m_damping << "\n"
                  << "  Maximum distance for SOM update = " << m_max_update_distance << "\n"
                  << "  Use periodic boundary conditions = " << m_use_pbc << "\n"
                  << "  Random shuffle data input = " << m_shuffle_data_input << "\n";
    } else if (m_executionPath == ExecutionPath::MAP) {
        std::cout << "  Store best rotation and flipping parameters = " << m_write_rot_flip << "\n";

        if (!m_rot_flip_filename.empty())
            std::cout << "  Best rotation and flipping parameter filename = " << m_rot_flip_filename << "\n";
    }

    if (m_verbose)
        std::cout << "  Block size 1 = " << m_block_size_1 << "\n";

    std::cout << std::endl;
}

void InputData::print_usage() const
{
    print_header();
    std::cout << "\n"
                 "  Usage:\n"
                 "\n"
                 "    Pink [Options] --train <image-file> <result-file>\n"
                 "    Pink [Options] --map   <image-file> <result-file> <SOM-file>\n"
                 "\n"
                 "  Options:\n"
                 "\n"
                 "    --cuda-off                                    "
                 "Switch off CUDA acceleration.\n"
                 "    --dist-func, -f <string>                      "
                 "Distribution function for SOM update (see below).\n"
                 "    --euclidean-distance-dimension, -e <int>      "
                 "Dimension for euclidean distance calculation (default = image-dimension * sqrt(2) / 2).\n"
                 "    --euclidean-distance-type                     "
                 "Data type for euclidean distance calculation (unit8 = default, uint16, float).\n"
                 "    --euclidean-distance-shape                    "
                 "Shape of euclidean distance region (quadratic = default, circular).\n"
                 "    --flip-off                                    "
                 "Switch off usage of mirrored images.\n"
                 "    --help, -h                                    "
                 "Print this lines.\n"
                 "    --init, -x <string>                           "
                 "Type of SOM initialization (zero = default, random, random_with_preferred_direction, file_init).\n"
                 "    --input-shuffle-off                           "
                 "Switch off random shuffle of data input (only for training).\n"
                 "    --interpolation <string>                      "
                 "Type of image interpolation for rotations (nearest_neighbor, bilinear = default).\n"
                 "    --inter-store <string>                        "
                 "Store intermediate SOM results at every progress step (off = default, overwrite, keep).\n"
                 "    --layout, -l <string>                         "
                 "Layout of SOM (cartesian = default, hexagonal).\n"
                 "    --max-update-distance <float>                 "
                 "Maximum distance for SOM update (default = off).\n"
                 "    --neuron-dimension, -d <int>                  "
                 "Dimension for quadratic SOM neurons (default = 2 * image-dimension / sqrt(2)).\n"
                 "    --numrot, -n <int>                            "
                 "Number of rotations (1 or a multiple of 4, default = 360).\n"
                 "    --numthreads, -t <int>                        "
                 "Number of CPU threads (default = auto).\n"
                 "    --num-iter <int>                              "
                 "Number of iterations (default = 1).\n"
                 "    --pbc                                         "
                 "Use periodic boundary conditions for SOM.\n"
                 "    --progress, -p <int>                          "
                 "Maximal number of progress information prints (default = 10).\n"
                 "    --seed, -s <unsigned int>                     "
                 "Seed for random number generator (default = 1234).\n"
                 "    --store-rot-flip <string>                     "
                 "Store the rotation and flip information of the best match of mapping.\n"
                 "    --som-width <int>                             "
                 "Width dimension of SOM (default = 10).\n"
                 "    --som-height <int>                            "
                 "Height dimension of SOM (default = 10).\n"
                 "    --som-depth <int>                             "
                 "Depth dimension of SOM (default = 1).\n"
                 "    --verbose                                     "
                 "Print more output.\n"
                 "    --version, -v                                 "
                 "Print version number.\n"
                 "\n"
                 "  Distribution function:\n"
                 "\n"
                 "    <string> <float> <float>\n"
                 "\n"
                 "    gaussian sigma damping-factor\n"
                 "    unitygaussian sigma damping-factor\n"
                 "    mexicanHat sigma damping-factor\n"
              << std::endl;
}

std::function<float(float)> InputData::get_distribution_function() const
{
    std::function<float(float)> result;
    if (m_distribution_function == DistributionFunction::GAUSSIAN)
        result = GaussianFunctor(m_sigma, m_damping);
    else if (m_distribution_function == DistributionFunction::UNITYGAUSSIAN)
        result = UnityGaussianFunctor(m_sigma, m_damping);
    else if (m_distribution_function == DistributionFunction::MEXICANHAT)
        result = MexicanHatFunctor(m_sigma, m_damping);
    else
        pink::exception("Unknown distribution function");
    return result;
}

} // namespace pink
