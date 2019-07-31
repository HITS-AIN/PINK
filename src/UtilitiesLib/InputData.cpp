/**
 * @file   InputData.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string.h>
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

} // end anonymous namespace

namespace pink {

InputData::InputData()
 : verbose(false),
   som_width(10),
   som_height(10),
   som_depth(1),
   neuron_dim(0),
   euclidean_distance_dim(0),
   layout(Layout::CARTESIAN),
   seed(1234),
   number_of_rotations(360),
   number_of_threads(-1),
   init(SOMInitialization::ZERO),
   number_of_iterations(1),
   max_number_of_progress_prints(10),
   use_flip(true),
   use_gpu(true),
   number_of_data_entries(0),
   data_layout(Layout::CARTESIAN),
   som_size(0),
   neuron_size(0),
   som_total_size(0),
   number_of_spatial_transformations(0),
   interpolation(Interpolation::BILINEAR),
   executionPath(ExecutionPath::UNDEFINED),
   intermediate_storage(IntermediateStorageType::OFF),
   distribution_function(DistributionFunction::GAUSSIAN),
   sigma(1.1f),
   damping(0.2f),
   block_size_1(256),
   max_update_distance(-1.0),
   usePBC(false),
   dimensionality(1),
   write_rot_flip(false),
   euclidean_distance_type(DataType::UINT8),
   shuffle_data_input(true)
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
        {NULL, 0, NULL, 0}
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
                neuron_dim = str_to_uint32_t(optarg);
                break;
            }
            case 'e':
            {
                euclidean_distance_dim = str_to_uint32_t(optarg);
                break;
            }
            case 0:
            {
                som_width = str_to_uint32_t(optarg);
                break;
            }
            case 12:
            {
                som_height = str_to_uint32_t(optarg);
                break;
            }
            case 13:
            {
                som_depth = str_to_uint32_t(optarg);
                break;
            }
            case 1:
            {
                number_of_iterations = str_to_uint32_t(optarg);
                if (number_of_iterations == 0)
                	throw pink::exception("Number of iterations must be larger than 0");
                break;
            }
            case 'l':
            {
                stringToUpper(optarg);
                if (strcmp(optarg, "CARTESIAN") == 0) layout = Layout::CARTESIAN;
                else if (strcmp(optarg, "HEXAGONAL") == 0) layout = Layout::HEXAGONAL;
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unknown option %o\n", c);
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 's':
            {
                seed = atoi(optarg);
                break;
            }
            case 'p':
            {
                max_number_of_progress_prints = std::atoi(optarg);
                break;
            }
            case 'n':
            {
                number_of_rotations = str_to_uint32_t(optarg);
                if (number_of_rotations == 0 or (number_of_rotations != 1 and number_of_rotations % 4))
                	throw pink::exception("Number of rotations must be 1 or a multiple of 4");
                break;
            }
            case 't':
            {
                number_of_threads = atoi(optarg);
                break;
            }
            case 'x':
            {
                char* upper_optarg = strdup(optarg);
                stringToUpper(upper_optarg);
                if (strcmp(upper_optarg, "ZERO") == 0) {
                    init = SOMInitialization::ZERO;
                } else if (strcmp(upper_optarg, "RANDOM") == 0) {
                    init = SOMInitialization::RANDOM;
                } else if (strcmp(upper_optarg, "RANDOM_WITH_PREFERRED_DIRECTION") == 0) {
                    init = SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION;
                } else {
                    init = SOMInitialization::FILEINIT;
                    som_filename = optarg;
                }
                break;
            }
            case 2:
            {
                use_flip = false;
                break;
            }
            case 3:
            {
                use_gpu = false;
                break;
            }
            case 4:
            {
                verbose = true;
                break;
            }
            case 5:
            {
                stringToUpper(optarg);
                if (strcmp(optarg, "NEAREST_NEIGHBOR") == 0) interpolation = Interpolation::NEAREST_NEIGHBOR;
                else if (strcmp(optarg, "BILINEAR") == 0) interpolation = Interpolation::BILINEAR;
                else {
                    print_usage();
                    printf ("optarg = %s\n", optarg);
                    printf ("Unknown option %o\n", c);
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 6:
            {
                executionPath = ExecutionPath::TRAIN;
                int index = optind - 1;
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --train option.");
                }
                data_filename = strdup(argv[index++]);
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --train option.");
                }
                result_filename = strdup(argv[index++]);
                optind = index - 1;
                break;
            }
            case 7:
            {
                executionPath = ExecutionPath::MAP;
                int index = optind - 1;
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --map option.");
                }
                data_filename = strdup(argv[index++]);
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --map option.");
                }
                result_filename = strdup(argv[index++]);
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --map option.");
                }
                som_filename = strdup(argv[index++]);
                optind = index - 1;
                break;
            }
            case 8:
            {
                stringToUpper(optarg);
                if (strcmp(optarg, "OFF") == 0) intermediate_storage = IntermediateStorageType::OFF;
                else if (strcmp(optarg, "OVERWRITE") == 0) intermediate_storage = IntermediateStorageType::OVERWRITE;
                else if (strcmp(optarg, "KEEP") == 0) intermediate_storage = IntermediateStorageType::KEEP;
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unknown option %o\n", c);
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 9:
            {
                block_size_1 = atoi(optarg);
                break;
            }
            case 10:
            {
                max_update_distance = std::strtof(optarg, &end_char);
                if (max_update_distance <= 0.0) {
                    print_usage();
                    throw pink::exception("max-update-distance must be positive.");
                }
                break;
            }
            case 14:
            {
                usePBC = true;
                break;
            }
            case 15:
            {
                write_rot_flip = true;
                rot_flip_filename = optarg;
                break;
            }
            case 16:
            {
                stringToUpper(optarg);
                if (strcmp(optarg, "FLOAT") == 0) euclidean_distance_type = DataType::FLOAT;
                else if (strcmp(optarg, "UINT16") == 0) euclidean_distance_type = DataType::UINT16;
                else if (strcmp(optarg, "UINT8") == 0) euclidean_distance_type = DataType::UINT8;
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unknown option %o\n", c);
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 17:
            {
                shuffle_data_input = false;
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
                stringToUpper(optarg);
                if (strcmp(optarg, "GAUSSIAN") == 0) {
                    distribution_function = DistributionFunction::GAUSSIAN;
                }
                else if (strcmp(optarg, "MEXICANHAT") == 0) {
                    distribution_function = DistributionFunction::MEXICANHAT;
                }
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unknown option %o\n", c);
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                int index = optind;
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --dist-func option.");
                }
                sigma = std::strtof(argv[index++], &end_char);
                if (index >= argc or argv[index][0] == '-') {
                    throw pink::exception("Missing arguments for --dist-func option.");
                }
                damping = std::strtof(argv[index++], &end_char);
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

    if (executionPath == ExecutionPath::MAP) {
        init = SOMInitialization::FILEINIT;
    } else if (executionPath == ExecutionPath::UNDEFINED) {
        print_usage();
        throw pink::exception("Unknown execution path.");
    }

    if (layout == Layout::HEXAGONAL) {
        if (usePBC) throw pink::exception("Periodic boundary conditions are not supported for hexagonal layout.");
        if ((som_width - 1) % 2) throw pink::exception("For hexagonal layout only odd dimension supported.");
        if (som_width != som_height) {
            throw pink::exception("For hexagonal layout som-width must be equal to som-height.");
        }
        if (som_depth != 1) throw pink::exception("For hexagonal layout som-depth must be equal to 1.");
        som_size = HexagonalLayout({som_width, som_height}).size();
    }
    else som_size = som_width * som_height * som_depth;

    if (som_width < 2) throw pink::exception("som-width must be > 1.");
    if (som_height < 1) throw pink::exception("som-height must be > 0.");
    if (som_depth < 1) throw pink::exception("som-depth must be > 0.");
    if (som_height > 1) ++dimensionality;
    if (som_depth > 1) ++dimensionality;

    std::ifstream ifs(data_filename);
    if (!ifs) throw std::runtime_error("Error opening " + data_filename);

    // Skip header
    get_file_header(ifs);

    int file_version, file_type, data_type;
    // Ignore first three entries
    ifs.read((char*)&file_version, sizeof(int));
    ifs.read((char*)&file_type, sizeof(int));
    ifs.read((char*)&data_type, sizeof(int));
    ifs.read((char*)&number_of_data_entries, sizeof(int));
    ifs.read((char*)&data_layout, sizeof(int));

    int data_dimensionality;
    ifs.read((char*)&data_dimensionality, sizeof(int));
    data_dimension.resize(static_cast<size_t>(data_dimensionality));

    for (size_t i = 0; i < static_cast<size_t>(data_dimensionality); ++i) {
        ifs.read((char*)&data_dimension[i], sizeof(int));
    }

    if (neuron_dim == 0) {
        neuron_dim = data_dimension[0];
        if (number_of_rotations != 1) neuron_dim = 2 * data_dimension[0]
						                         / static_cast<uint32_t>(std::sqrt(2.0)) + 1;
    }

    if (euclidean_distance_dim == 0) {
        euclidean_distance_dim = data_dimension[0];
        if (number_of_rotations != 1) euclidean_distance_dim *= static_cast<uint32_t>(std::sqrt(2.0)) / 2;
    }

    neuron_size = neuron_dim * neuron_dim;
    som_total_size = som_size * neuron_size;
    number_of_spatial_transformations = use_flip ? 2 * number_of_rotations : number_of_rotations;

    if (number_of_threads == -1) number_of_threads = omp_get_max_threads();
    omp_set_num_threads(number_of_threads);

    print_header();
    print_parameters();

    if (file_version != 2) throw pink::exception("Please use file format version 2 as data input.");
    if (file_type != 0) throw pink::exception("Please use file type 0 as data input.");
    if (data_type != 0) throw pink::exception("Only data_type = 0 (float, single precision) is supported.");
    if (number_of_data_entries <= 0) throw pink::exception("Number of data entries must be larger than 0.");
    if (euclidean_distance_dim > neuron_dim)
        throw pink::exception("euclidean distance dimension must be equal or smaller than neuron dimension.");
    if (usePBC) throw pink::exception("Periodic boundary conditions are not supported in version 2.");
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
    std::cout << "  Data file = " << data_filename << "\n"
              << "  Result file = " << result_filename << "\n";

    if (executionPath == ExecutionPath::MAP)
        std::cout << "  SOM file = " << som_filename << "\n";

    std::cout << "  Number of data entries = " << number_of_data_entries << "\n"
              << "  Data dimension = " << data_dimension[0];

    for (size_t i = 1; i < data_dimension.size(); ++i) std::cout << " x " << data_dimension[i];
    std::cout << std::endl;

    std::cout << "  SOM dimension (width x height x depth) = "
              << som_width << "x" << som_height << "x" << som_depth << "\n"
              << "  SOM size = " << som_size << "\n"
              << "  Number of iterations = " << number_of_iterations << "\n"
              << "  Neuron dimension = " << neuron_dim << "x" << neuron_dim << "\n"
              << "  Euclidean distance dimension = " << euclidean_distance_dim << "x" << euclidean_distance_dim << "\n"
              << "  Maximal number of progress information prints = " << max_number_of_progress_prints << "\n"
              << "  Intermediate storage of SOM = " << intermediate_storage << "\n"
              << "  Layout = " << layout << "\n"
              << "  Initialization type = " << init;

    if (init == SOMInitialization::FILEINIT) std::cout << "\n  SOM initialization file = " << som_filename;

    std::cout << "\n"
              << "  Interpolation type = " << interpolation << "\n"
              << "  Seed = " << seed << "\n"
              << "  Number of rotations = " << number_of_rotations << "\n"
              << "  Use mirrored image = " << use_flip << "\n"
              << "  Number of CPU threads = " << number_of_threads << "\n"
              << "  Use CUDA = " << use_gpu << "\n";

    if (executionPath == ExecutionPath::TRAIN) {
        std::cout << "  Distribution function for SOM update = " << distribution_function << "\n"
                  << "  Sigma = " << sigma << "\n"
                  << "  Damping factor = " << damping << "\n"
                  << "  Maximum distance for SOM update = " << max_update_distance << "\n"
                  << "  Use periodic boundary conditions = " << usePBC << "\n"
                  << "  Random shuffle data input = " << shuffle_data_input << "\n";
    } else if (executionPath == ExecutionPath::MAP) {
        std::cout << "  Store best rotation and flipping parameters = " << write_rot_flip << "\n";

        if (!rot_flip_filename.empty())
            std::cout << "  Best rotation and flipping parameter filename = " << rot_flip_filename << "\n";
    }

    if (verbose)
        std::cout << "  Block size 1 = " << block_size_1 << "\n";

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
                 "    --seed, -s <int>                              "
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
                 "    mexicanHat sigma damping-factor\n"
              << std::endl;
}

std::function<float(float)> InputData::get_distribution_function() const
{
    std::function<float(float)> result;
    if (distribution_function == DistributionFunction::GAUSSIAN)
        result = GaussianFunctor(sigma, damping);
    else if (distribution_function == DistributionFunction::MEXICANHAT)
        result = MexicanHatFunctor(sigma, damping);
    else
        pink::exception("Unknown distribution function");
    return result;
}

void stringToUpper(char* s)
{
    for (char *ps = s; *ps != '\0'; ++ps) *ps = toupper(*ps);
}

} // namespace pink
