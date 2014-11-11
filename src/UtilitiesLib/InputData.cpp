/**
 * @file   InputData.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "InputData.h"
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <omp.h>
#include <string.h>
#include <sstream>
#include <stdlib.h>

using namespace std;

InputData::InputData(int argc, char **argv)
 :
	verbose(true),
	imagesFilename(),
	som_dim(10),
	neuron_dim(-1),
	layout(QUADRATIC),
	resultFilename(),
	seed(1234),
	numberOfRotations(360),
	numberOfThreads(-1),
	init(ZERO),
	numIter(1),
	progressFactor(0.1),
	useFlip(true),
	useCuda(true),
    numberOfImages(0),
    image_dim(0),
    image_size(0),
    som_size(0),
    neuron_size(0),
    som_total_size(0),
    numberOfRotationsAndFlip(0),
    algo(0)
{
	static struct option long_options[] = {
		{"image-file",      1, 0, 'i'},
		{"image-dimension", 1, 0, 'd'},
		{"som-dimension",   1, 0, 0},
		{"result-file",     1, 0, 'r'},
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
		{NULL, 0, NULL, 0}
	};
	int c, option_index = 0;
	while ((c = getopt_long(argc, argv, "vi:d:r:l:s:n:t:x:p:a:", long_options, &option_index)) != -1)
	{
		int this_option_optind = optind ? optind : 1;
		switch (c) {
		case 4:
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
		case 'i':
			imagesFilename = optarg;
			break;
		case 'd':
			neuron_dim = atoi(optarg);
			break;
		case 'a':
			algo = atoi(optarg);
			break;
		case 0:
			som_dim = atoi(optarg);
			break;
		case 1:
			numIter = atoi(optarg);
			break;
		case 'r':
			resultFilename = optarg;
			break;
		case 'l':
			stringToUpper(optarg);
			if (strcmp(optarg, "QUADRATIC") == 0) layout = QUADRATIC;
			else if (strcmp(optarg, "HEXAGONAL") == 0) layout = HEXAGONAL;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 's':
			seed = atoi(optarg);
			break;
		case 'p':
			progressFactor = atof(optarg);
			break;
		case 'n':
			numberOfRotations = atoi(optarg);
			if (numberOfRotations < 0) {
				printf ("Number of rotations must be larger than 0.\n\n");
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 't':
			numberOfThreads = atoi(optarg);
			if (useCuda and numberOfThreads > 1) {
				printf ("Number of CPU threads must be 1 using CUDA.\n\n");
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 'x':
			stringToUpper(optarg);
			if (strcmp(optarg, "ZERO") == 0) init = ZERO;
			else if (strcmp(optarg, "RANDOM") == 0) init = RANDOM;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				exit(EXIT_FAILURE);
			}
			break;
		case 2:
			useFlip = false;
			break;
		case 3:
			useCuda = false;
			break;
		case 'v':
			cout << "Pink version " << PINK_VERSION_MAJOR << "." << PINK_VERSION_MINOR << endl;
			exit(0);
		case '?':
			printf ("Unkown option %o\n", c);
			print_usage();
			exit(EXIT_FAILURE);
		default:
			printf ("Unkown option %o\n", c);
			print_usage();
			exit(EXIT_FAILURE);
		}
	}

	if (optind < argc) {
		cout << "Unkown argv elements: ";
		while (optind < argc) cout << argv[optind++] << " ";
		cout << endl;
		print_usage();
		exit(EXIT_FAILURE);
	}

	// Check if all non-optional arguments are set
	if (imagesFilename.empty() or resultFilename.empty()) {
		cout << "Missing non-optional argument." << endl;
		print_usage();
		exit(EXIT_FAILURE);
	}

	PINK::ImageIterator<float> iterImage(imagesFilename);
	if (iterImage->getWidth() != iterImage->getHeight()) {
		cout << "Only quadratic images are supported.";
		exit(EXIT_FAILURE);
	}

	numberOfImages = iterImage.number();
	image_dim = iterImage->getWidth();
	image_size = image_dim * image_dim;
    som_size = som_dim * som_dim;

    if (neuron_dim == -1) neuron_dim = image_dim * sqrt(2.0) / 2.0;
    if (neuron_dim > image_dim) {
		cout << "Neuron dimension must be smaller or equal to image dimension.";
		exit(EXIT_FAILURE);
    }
    if ((image_dim - neuron_dim)%2) --neuron_dim;

    neuron_size = neuron_dim * neuron_dim;
	som_total_size = som_size * neuron_size;

    numberOfRotationsAndFlip = useFlip ? 2*numberOfRotations : numberOfRotations;

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
		     << "  Number of images = " << numberOfImages << endl
		     << "  Image dimension = " << image_dim << "x" << image_dim << endl
		     << "  SOM dimension = " << som_dim << "x" << som_dim << endl
		     << "  Number of iterations = " << numIter << endl
		     << "  Neuron dimension = " << neuron_dim << "x" << neuron_dim << endl
		     << "  Progress = " << progressFactor << endl
		     << "  Result file = " << resultFilename << endl
		     << "  Layout = " << layout << endl
		     << "  Initialization type = " << init << endl
		     << "  Seed = " << seed << endl
		     << "  Number of rotations = " << numberOfRotations << endl
		     << "  Use mirrored image = " << useFlip << endl
		     << "  Number of CPU threads = " << numberOfThreads << endl
		     << "  Use CUDA = " << useCuda << endl
	         << "  CUDA algorithm = " << algo << endl;
	}
}

void InputData::print_usage() const
{
    print_header();
	cout << "\n"
	        "  USAGE: Pink -i <image-file> -r <result-file>\n"
			"\n"
	        "  Non-optional options:\n"
			"\n"
	        "    --image-file, -i        File with images.\n"
	        "    --result-file, -r       File for final SOM matrix.\n"
			"\n"
	        "  Optional options:\n"
			"\n"
	        "    --algo, -a              Specific GPU algorithm (default = 0).\n"
			"                            0: FindBestNeuron on GPU, ImageRotation and UpdateSOM on CPU\n"
			"                            1: ImageRotation and FindBestNeuron on GPU, UpdateSOM on CPU\n"
	        "    --som-dimension         Dimension for quadratic SOM matrix (default = 10).\n"
	        "    --neuron-dimension, -d  Dimension for quadratic SOM neurons (default = image-size * sqrt(2)/2).\n"
	        "    --num-iter              Number of iterations (default = 1).\n"
	        "    --layout, -l            Layout of SOM (quadratic, hexagonal, default = quadratic).\n"
	        "    --seed, -s              Seed for random number generator (default = 1234).\n"
			"    --progress, -p          Print level of progress (default = 10%).\n"
	        "    --numrot, -n            Number of rotations (default = 360).\n"
	        "    --flip-off              Switch off usage of mirrored images (default = on).\n"
	        "    --numthreads, -t        Number of CPU threads (default = auto).\n"
	        "    --init, -x              Type of SOM initialization (random, zero, default = zero).\n"
            "    --cuda-off              Switch off CUDA acceleration (default = on).\n"
            "    --version, -v           Print version number.\n"
            "    --verbose               Print more output (yes, no, default = yes).\n" << endl;
}

void stringToUpper(char* s)
{
	for (char *ps = s; *ps != '\0'; ++ps) *ps = toupper(*ps);
}
