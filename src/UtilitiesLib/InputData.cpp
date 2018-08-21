/**
 * @file   InputData.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "InputData.h"
#include "UtilitiesLib/Error.h"
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <omp.h>
#include <string.h>
#include <sstream>
#include <stdlib.h>

using namespace std;

std::ostream& operator << (std::ostream& os, Layout layout)
{
    if (layout == QUADRATIC) os << "quadratic";
    else if (layout == HEXAGONAL) os << "hexagonal";
    else if (layout == QUADHEX) os << "quadhex";
    else os << "undefined";
    return os;
}

std::ostream& operator << (std::ostream& os, Function function)
{
    if (function == GAUSSIAN) os << "gaussian";
    else if (function == MEXICANHAT) os << "mexicanhat";
    else os << "undefined";
    return os;
}

std::ostream& operator << (std::ostream& os, SOMInitialization init)
{
    if (init == ZERO) os << "zero";
    else if (init == RANDOM) os << "random";
    else if (init == RANDOM_WITH_PREFERRED_DIRECTION) os << "random_with_preferred_direction";
    else if (init == FILEINIT) os << "file_init";
    else os << "undefined";
    return os;
}

std::ostream& operator << (std::ostream& os, IntermediateStorageType type)
{
    if (type == OFF) os << "off";
    else if (type == OVERWRITE) os << "overwrite";
    else if (type == KEEP) os << "keep";
    else os << "undefined";
    return os;
}

InputData::InputData()
 :
    verbose(false),
    som_width(10),
    som_height(10),
    som_depth(1),
    neuron_dim(-1),
    layout(QUADRATIC),
    seed(1234),
    numberOfRotations(360),
    numberOfThreads(-1),
    init(ZERO),
    numIter(1),
    progressFactor(0.1),
    useFlip(true),
    useCuda(true),
    numberOfImages(0),
    numberOfChannels(0),
    image_dim(0),
    image_size(0),
    som_size(0),
    neuron_size(0),
    som_total_size(0),
    numberOfRotationsAndFlip(0),
    interpolation(BILINEAR),
    executionPath(UNDEFINED),
    intermediate_storage(OFF),
    function(GAUSSIAN),
    sigma(DEFAULT_SIGMA),
    damping(DEFAULT_DAMPING),
    block_size_1(256),
    maxUpdateDistance(-1.0),
    useMultipleGPUs(true),
    usePBC(false),
    dimensionality(1)
{}

InputData::InputData(int argc, char **argv)
 : InputData()
{
    static struct option long_options[] = {
        {"neuron-dimension",    1, 0, 'd'},
        {"layout",              1, 0, 'l'},
        {"seed",                1, 0, 's'},
        {"numrot",              1, 0, 'n'},
        {"numthreads",          1, 0, 't'},
        {"init",                1, 0, 'x'},
        {"progress",            1, 0, 'p'},
        {"version",             0, 0, 'v'},
        {"help",                0, 0, 'h'},
        {"dist-func",           1, 0, 'f'},
        {"som-width",           1, 0, 0},
        {"num-iter",            1, 0, 1},
        {"flip-off",            0, 0, 2},
        {"cuda-off",            0, 0, 3},
        {"verbose",             0, 0, 4},
        {"interpolation",       1, 0, 5},
        {"train",               1, 0, 6},
        {"map",                 1, 0, 7},
        {"inter-store",         1, 0, 8},
        {"b1",                  1, 0, 9},
        {"max-update-distance", 1, 0, 10},
        {"multi-GPU-off",       0, 0, 11},
        {"som-height",          1, 0, 12},
        {"som-depth",           1, 0, 13},
        {"pbc",                 0, 0, 14},
        {NULL, 0, NULL, 0}
    };
    int c, option_index = 0;
    while ((c = getopt_long(argc, argv, "vd:l:s:n:t:x:p:a:hf:", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'd':
            {
                neuron_dim = atoi(optarg);
                break;
            }
            case 0:
            {
                som_width = atoi(optarg);
                break;
            }
            case 12:
            {
                som_height = atoi(optarg);
                break;
            }
            case 13:
            {
                som_depth = atoi(optarg);
                break;
            }
            case 1:
            {
                numIter = atoi(optarg);
                if (numIter < 0) {
                    print_usage();
                    printf ("ERROR: Number of iterations must be larger than 0.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'l':
            {
                stringToUpper(optarg);
                if (strcmp(optarg, "QUADRATIC") == 0) layout = QUADRATIC;
                else if (strcmp(optarg, "HEXAGONAL") == 0) layout = HEXAGONAL;
                else if (strcmp(optarg, "QUADHEX") == 0) layout = QUADHEX;
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unkown option %o\n", c);
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
                progressFactor = atof(optarg);
                break;
            }
            case 'n':
            {
                numberOfRotations = atoi(optarg);
                if (numberOfRotations <= 0 or (numberOfRotations != 1 and numberOfRotations % 4)) {
                    print_usage();
                    printf ("ERROR: Number of rotations must be 1 or a multiple of 4.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 't':
            {
                numberOfThreads = atoi(optarg);
                if (useCuda and numberOfThreads > 1) {
                    print_usage();
                    printf ("ERROR: Number of CPU threads must be 1 using CUDA.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'x':
            {
                char* upper_optarg = strdup(optarg);
                stringToUpper(upper_optarg);
                if (strcmp(upper_optarg, "ZERO") == 0) init = ZERO;
                else if (strcmp(upper_optarg, "RANDOM") == 0) init = RANDOM;
                else if (strcmp(upper_optarg, "RANDOM_WITH_PREFERRED_DIRECTION") == 0) init = RANDOM_WITH_PREFERRED_DIRECTION;
                else {
                    init = FILEINIT;
                    somFilename = optarg;
                }
                break;
            }
            case 2:
            {
                useFlip = false;
                break;
            }
            case 3:
            {
                useCuda = false;
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
                if (strcmp(optarg, "NEAREST_NEIGHBOR") == 0) interpolation = NEAREST_NEIGHBOR;
                else if (strcmp(optarg, "BILINEAR") == 0) interpolation = BILINEAR;
                else {
                    print_usage();
                    printf ("optarg = %s\n", optarg);
                    printf ("Unkown option %o\n", c);
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 6:
            {
                executionPath = TRAIN;
                int index = optind - 1;
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --train option.");
                imagesFilename = strdup(argv[index++]);
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --train option.");
                resultFilename = strdup(argv[index++]);
                optind = index - 1;
                break;
            }
            case 7:
            {
                executionPath = MAP;
                int index = optind - 1;
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --map option.");
                imagesFilename = strdup(argv[index++]);
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --map option.");
                resultFilename = strdup(argv[index++]);
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --map option.");
                somFilename = strdup(argv[index++]);
                optind = index - 1;
                break;
            }
            case 8:
            {
                stringToUpper(optarg);
                if (strcmp(optarg, "OFF") == 0) intermediate_storage = OFF;
                else if (strcmp(optarg, "OVERWRITE") == 0) intermediate_storage = OVERWRITE;
                else if (strcmp(optarg, "KEEP") == 0) intermediate_storage = KEEP;
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unkown option %o\n", c);
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
                maxUpdateDistance = atof(optarg);
                if (maxUpdateDistance <= 0.0) {
                    print_usage();
                    fatalError("max-update-distance must be positive.");
                }
                break;
            }
            case 11:
            {
                useMultipleGPUs = false;
                break;
            }
            case 14:
            {
                usePBC = true;
                break;
            }
            case 'v':
            {
                cout << "Pink version " << PROJECT_VERSION << endl;
                cout << "Git revision " << GIT_REVISION << endl;
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
                    function = GAUSSIAN;
                }
                else if (strcmp(optarg, "MEXICANHAT") == 0) {
                    function = MEXICANHAT;
                }
                else {
                    printf ("optarg = %s\n", optarg);
                    printf ("Unkown option %o\n", c);
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                int index = optind;
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --dist-func option.");
                sigma = atof(argv[index++]);
                if (index >= argc or argv[index][0] == '-') fatalError("Missing arguments for --dist-func option.");
                damping = atof(argv[index++]);
                optind = index;
                break;
            }
            case '?':
            {
                printf ("Unkown option %o\n", c);
                print_usage();
                exit(EXIT_FAILURE);
            }
            default:
            {
                printf ("Unkown option %o\n", c);
                print_usage();
                exit(EXIT_FAILURE);
            }
        }
    }

    if (executionPath == MAP) {
        init = FILEINIT;
    } else if (executionPath == UNDEFINED) {
        print_usage();
        fatalError("Unkown execution path.");
    }

    PINK::ImageIterator<float> iterImage(imagesFilename);

    if (iterImage->getWidth() != iterImage->getHeight()) {
        print_usage();
        fatalError("Only quadratic images are supported.");
    }

    numberOfImages = iterImage.getNumberOfImages();
    numberOfChannels = iterImage.getNumberOfChannels();
    image_dim = iterImage->getWidth();
    image_size = image_dim * image_dim;

    if (layout == HEXAGONAL) {
        if (usePBC) fatalError("Periodic boundary conditions are not supported for hexagonal layout.");
        if ((som_width - 1) % 2) fatalError("For hexagonal layout only odd dimension supported.");
        if (som_width != som_height) fatalError("For hexagonal layout som-width must be equal to som-height.");
        if (som_depth != 1) fatalError("For hexagonal layout som-depth must be equal to 1.");
        int radius = (som_width - 1)/2;
        som_size = som_width * som_height - radius * (radius + 1);
    }
    else som_size = som_width * som_height * som_depth;

    if (som_width < 2) fatalError("som-width must be > 1.");
    if (som_height < 1) fatalError("som-height must be > 0.");
    if (som_depth < 1) fatalError("som-depth must be > 0.");
    if (som_height > 1) ++dimensionality;
    if (som_depth > 1) ++dimensionality;

    if (neuron_dim == -1) neuron_dim = image_dim * sqrt(2.0) / 2.0;
    if (neuron_dim > image_dim) {
        print_usage();
        cout << "ERROR: Neuron dimension must be smaller or equal to image dimension.";
        exit(EXIT_FAILURE);
    }

    neuron_size = neuron_dim * neuron_dim;
    som_total_size = som_size * neuron_size;
    numberOfRotationsAndFlip = useFlip ? 2*numberOfRotations : numberOfRotations;

    if (numberOfThreads == -1) numberOfThreads = omp_get_num_procs();
#if PINK_USE_CUDA
    if (useCuda) numberOfThreads = 1;
#endif
    omp_set_num_threads(numberOfThreads);

    print_header();
    print_parameters();
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
                 "  *     Parallel orientation Invariant Non-parametric Kohonen-map         *\n"
                 "  *                                                                       *\n"
                 "  *                         Version " << PROJECT_VERSION << "                                   *\n"
                 "  *                         Git revision: " << GIT_REVISION << "                         *\n"
                 "  *                                                                       *\n"
                 "  *     Bernd Doser <bernd.doser@h-its.org>                               *\n"
                 "  *     Kai Polsterer <kai.polsterer@h-its.org>                           *\n"
                 "  *                                                                       *\n"
                 "  *     Distributed under the GNU GPLv3 License.                          *\n"
                 "  *     See accompanying file LICENSE or                                  *\n"
                 "  *     copy at http://www.gnu.org/licenses/gpl-3.0.html.                 *\n"
                 "  *                                                                       *\n"
                 "  *************************************************************************\n"
              << std::endl;
}

void InputData::print_parameters() const
{
    cout << "  Image file = " << imagesFilename << endl
         << "  Result file = " << resultFilename << endl;

    if (executionPath == MAP)
        cout << "  SOM file = " << somFilename << endl;

    cout << "  Number of images = " << numberOfImages << endl
         << "  Number of channels = " << numberOfChannels << endl
         << "  Image dimension = " << image_dim << "x" << image_dim << endl
         << "  SOM dimension (width x height x depth) = " << som_width << "x" << som_height << "x" << som_depth << endl
         << "  SOM size = " << som_size << endl
         << "  Number of iterations = " << numIter << endl
         << "  Neuron dimension = " << neuron_dim << "x" << neuron_dim << endl
         << "  Progress = " << progressFactor << endl
         << "  Intermediate storage of SOM = " << intermediate_storage << endl
         << "  Layout = " << layout << endl
         << "  Initialization type = " << init << endl
         << "  Interpolation type = " << interpolation << endl
         << "  Seed = " << seed << endl
         << "  Number of rotations = " << numberOfRotations << endl
         << "  Use mirrored image = " << useFlip << endl
         << "  Number of CPU threads = " << numberOfThreads << endl
         << "  Use CUDA = " << useCuda << endl
         << "  Use multiple GPUs = " << useMultipleGPUs << endl
         << "  Distribution function for SOM update = " << function << endl
         << "  Sigma = " << sigma << endl
         << "  Damping factor = " << damping << endl
         << "  Maximum distance for SOM update = " << maxUpdateDistance << endl
         << "  Use periodic boundary conditions = " << usePBC << endl
         << endl;

    if (verbose)
        cout << "  Block size 1 = " << block_size_1 << endl
             << endl;
}

void InputData::print_usage() const
{
    print_header();
    cout << "\n"
            "  Usage:\n"
            "\n"
            "    Pink [Options] --train <image-file> <result-file>\n"
            "    Pink [Options] --map   <image-file> <result-file> <SOM-file>\n"
            "\n"
            "  Options:\n"
            "\n"
            "    --cuda-off                      Switch off CUDA acceleration.\n"
            "    --dist-func, -f <string>        Distribution function for SOM update (see below).\n"
            "    --flip-off                      Switch off usage of mirrored images.\n"
            "    --help, -h                      Print this lines.\n"
            "    --init, -x <string>             Type of SOM initialization (zero = default, random, random_with_preferred_direction, SOM-file).\n"
            "    --interpolation <string>        Type of image interpolation for rotations (nearest_neighbor, bilinear = default).\n"
            "    --inter-store <string>          Store intermediate SOM results at every progress step (off = default, overwrite, keep).\n"
            "    --layout, -l <string>           Layout of SOM (quadratic = default, quadhex, hexagonal).\n"
            "    --neuron-dimension, -d <int>    Dimension for quadratic SOM neurons (default = image-dimension * sqrt(2)/2).\n"
            "    --numrot, -n <int>              Number of rotations (1 or a multiple of 4, default = 360).\n"
            "    --numthreads, -t <int>          Number of CPU threads (default = auto).\n"
            "    --num-iter <int>                Number of iterations (default = 1).\n"
            "    --multi-GPU-off                 Switch off usage of multiple GPUs.\n"
            "    --pbc                           Use periodic boundary conditions for SOM.\n"
            "    --progress, -p <float>          Print level of progress (default = 0.1).\n"
            "                                    If < 1 relative progress, else number of images.\n"
            "    --seed, -s <int>                Seed for random number generator (default = 1234).\n"
            "    --som-width <int>               Width dimension of SOM (default = 10).\n"
            "    --som-height <int>              Height dimension of SOM (default = 10).\n"
            "    --som-depth <int>               Depth dimension of SOM (default = 1).\n"
            "    --max-update-distance <float>   Maximum distance for SOM update (default = off).\n"
            "    --version, -v                   Print version number.\n"
            "    --verbose                       Print more output.\n"
            "\n"
            "  Distribution function:\n"
            "\n"
            "    <string> <float> <float>\n"
            "\n"
            "    gaussian sigma damping-factor\n"
            "    mexicanHat sigma damping-factor\n"
            << endl;
}

void stringToUpper(char* s)
{
    for (char *ps = s; *ps != '\0'; ++ps) *ps = toupper(*ps);
}
