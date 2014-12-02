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

InputData::InputData(int argc, char **argv)
 :
	verbose(true),
	som_dim(10),
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
    image_size_using_flip(0),
    som_size(0),
    neuron_size(0),
    som_total_size(0),
    numberOfRotationsAndFlip(0),
    algo(2),
    interpolation(BILINEAR),
    executionPath(UNDEFINED),
    intermediate_storage(false),
    function(GAUSSIAN),
    sigma(UPDATE_NEURONS_SIGMA),
    damping(UPDATE_NEURONS_DAMPING)
{
	static struct option long_options[] = {
		{"image-dimension", 1, 0, 'd'},
		{"som-dimension",   1, 0, 0},
		{"layout",          1, 0, 'l'},
		{"seed",            1, 0, 's'},
		{"numrot",          1, 0, 'n'},
		{"numthreads",      1, 0, 't'},
		{"init",            1, 0, 'x'},
		{"num-iter",        1, 0, 1},
		{"progress",        1, 0, 'p'},
		{"flip-off",        0, 0, 2},
		{"cuda-off",        0, 0, 3},
		{"verbose",         0, 0, 4},
		{"version",         0, 0, 'v'},
		{"algo",            1, 0, 'a'},
		{"help",            0, 0, 'h'},
		{"interpolation",   1, 0, 5},
		{"train",           1, 0, 6},
		{"map",             1, 0, 7},
        {"inter-store",     1, 0, 8},
        {"dist-func",       1, 0, 'f'},
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
			case 'a':
			{
				algo = atoi(optarg);
				break;
			}
			case 0:
			{
				som_dim = atoi(optarg);
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
				stringToUpper(optarg);
				if (strcmp(optarg, "YES") == 0) verbose = true;
				else if (strcmp(optarg, "NO") == 0) verbose = false;
				else {
					printf ("optarg = %s\n", optarg);
					printf ("Unkown option %o\n", c);
					print_usage();
					exit(EXIT_FAILURE);
				}
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
			case 'v':
			{
				cout << "Pink version " << PINK_VERSION_MAJOR << "." << PINK_VERSION_MINOR << endl;
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

	if (executionPath == UNDEFINED) {
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
    som_size = som_dim * som_dim;

    if (neuron_dim == -1) neuron_dim = image_dim * sqrt(2.0) / 2.0;
    if (neuron_dim > image_dim) {
		print_usage();
		cout << "ERROR: Neuron dimension must be smaller or equal to image dimension.";
		exit(EXIT_FAILURE);
    }
    if ((image_dim - neuron_dim)%2) --neuron_dim;

    neuron_size = neuron_dim * neuron_dim;
	som_total_size = som_size * neuron_size;

    numberOfRotationsAndFlip = useFlip ? 2*numberOfRotations : numberOfRotations;
	image_size_using_flip = useFlip ? 2*image_size : image_size;

	if (numberOfThreads == -1) numberOfThreads = omp_get_num_procs();
    if (useCuda) numberOfThreads = 1;
	omp_set_num_threads(numberOfThreads);

    print_header();
    print_parameters();
}

void InputData::print_header() const
{
	string rawHeader = "Parallel orientation Invariant Non-parametric Kohonen-map (PINK)";

	stringstream ssVersion;
	ssVersion << "Version " << PINK_VERSION_MAJOR << "." << PINK_VERSION_MINOR;

	int diff = rawHeader.size() - ssVersion.str().size();
	int leftBlanks = diff / 2;
	int rightBlanks = diff / 2;
	rightBlanks += diff % 2 ? 1 : 0;

	cout << "\n"
	        "  ************************************************************************\n"
	        "  *   Parallel orientation Invariant Non-parametric Kohonen-map (PINK)   *\n"
	        "  *                                                                      *\n"
	        "  *   " << string(leftBlanks,' ') << ssVersion.str() << string(rightBlanks,' ') << "   *\n"
	        "  *                                                                      *\n"
	        "  *   Kai Polsterer, Bernd Doser, HITS gGmbH                             *\n"
	        "  ************************************************************************\n" << endl;
}

void InputData::print_parameters() const
{
	if (verbose) {
		cout << "  Image file = " << imagesFilename << endl
			 << "  Result file = " << resultFilename << endl;
		if (executionPath == MAP) cout << "  SOM file = " << somFilename << endl;
		cout << "  Number of images = " << numberOfImages << endl
		     << "  Number of channels = " << numberOfChannels << endl
		     << "  Image dimension = " << image_dim << "x" << image_dim << endl
		     << "  SOM dimension = " << som_dim << "x" << som_dim << endl
		     << "  Number of iterations = " << numIter << endl
		     << "  Neuron dimension = " << neuron_dim << "x" << neuron_dim << endl
		     << "  Progress = " << progressFactor << endl
		     << "  Layout = " << layout << endl
		     << "  Initialization type = " << init << endl
		     << "  Interpolation type = " << interpolation << endl
		     << "  Seed = " << seed << endl
		     << "  Number of rotations = " << numberOfRotations << endl
		     << "  Use mirrored image = " << useFlip << endl
		     << "  Number of CPU threads = " << numberOfThreads << endl
		     << "  Use CUDA = " << useCuda << endl
             << "  CUDA algorithm = " << algo << endl
             << "  Distribution function for SOM update = " << function << endl
	         << "  Sigma = " << sigma << endl
             << "  Damping factor = " << damping << endl;
	}
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
	        "    --layout, -l <string>           Layout of SOM (quadratic = default, quadhex, hexagonal).\n"
	        "    --neuron-dimension, -d <int>    Dimension for quadratic SOM neurons (default = image-size * sqrt(2)/2).\n"
	        "    --numrot, -n <int>              Number of rotations (1 or a multiple of 4, default = 360).\n"
	        "    --numthreads, -t <int>          Number of CPU threads (default = auto).\n"
	        "    --num-iter <int>                Number of iterations (default = 1).\n"
			"    --progress, -p <float>          Print level of progress (default = 0.1).\n"
	        "    --seed, -s <int>                Seed for random number generator (default = 1234).\n"
            "    --inter-store                   Store intermediate SOM results at every progress step.\n"
	        "    --som-dimension <int>           Dimension for quadratic SOM matrix (default = 10).\n"
            "    --version, -v                   Print version number.\n"
            "    --verbose                       Print more output.\n"
	        "\n"
	        "  Distribution function string:\n"
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
