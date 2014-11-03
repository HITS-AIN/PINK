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
#include <string.h>
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
    neuron_size(0)
{
	static struct option long_options[] = {
		{"verbose",         0, 0, 'v'},
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
		{"flip",            1, 0, 2},
		{"cuda",            1, 0, 3},
		{NULL, 0, NULL, 0}
	};
	int c, option_index = 0;
	while ((c = getopt_long(argc, argv, "v:i:d:r:l:s:n:t:x:p:", long_options, &option_index)) != -1)
	{
		int this_option_optind = optind ? optind : 1;
		switch (c) {
		case 'v':
			stringToUpper(optarg);
			if (strcmp(optarg, "YES") == 0) verbose = true;
			else if (strcmp(optarg, "NO") == 0) verbose = false;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				exit(1);
			}
			break;
		case 'i':
			imagesFilename = optarg;
			break;
		case 'd':
			neuron_dim = atoi(optarg);
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
				exit(1);
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
				exit(1);
			}
			break;
		case 't':
			numberOfThreads = atoi(optarg);
			break;
		case 'x':
			stringToUpper(optarg);
			if (strcmp(optarg, "ZERO") == 0) init = ZERO;
			else if (strcmp(optarg, "RANDOM") == 0) init = RANDOM;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				exit(1);
			}
			break;
		case 2:
			stringToUpper(optarg);
			if (strcmp(optarg, "YES") == 0) useFlip = true;
			else if (strcmp(optarg, "NO") == 0) useFlip = false;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				exit(1);
			}
			break;
		case 3:
			stringToUpper(optarg);
			if (strcmp(optarg, "YES") == 0) useCuda = true;
			else if (strcmp(optarg, "NO") == 0) useCuda = false;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				exit(1);
			}
			break;
		case '?':
			printf ("Unkown option %o\n", c);
			print_usage();
			exit(1);
		default:
			printf ("Unkown option %o\n", c);
			print_usage();
			exit(1);
		}
	}

	if (optind < argc) {
		cout << "Unkown argv elements: ";
		while (optind < argc) cout << argv[optind++] << " ";
		cout << endl;
		print_usage();
		exit(1);
	}

	// Check if all non-optional arguments are set
	if (imagesFilename.empty() or resultFilename.empty()) {
		cout << "Missing non-optional argument." << endl;
		print_usage();
		exit(1);
	}

	PINK::ImageIterator<float> iterImage(imagesFilename);
	if (verbose) cout << "  Image dimension = " << iterImage->getWidth() << "x" << iterImage->getHeight() << endl;

	if (iterImage->getWidth() != iterImage->getHeight()) {
		cout << "Only quadratic images are supported.";
		exit(1);
	}

	numberOfImages = iterImage.number();
	image_dim = iterImage->getWidth();
	image_size = iterImage->getWidth() * iterImage->getHeight();
    som_size = som_dim * som_dim;

    if (neuron_dim == -1) neuron_dim = image_dim * sqrt(2.0) / 2.0;
    if (neuron_dim > image_dim) {
		cout << "Neuron dimension must be smaller or equal to image dimension.";
		exit(1);
    }
    if ((image_dim - neuron_dim)%2) --neuron_dim;
    neuron_size = neuron_dim * neuron_dim;
}

void InputData::print() const
{
	if (verbose) {
		cout << "  Image file = " << imagesFilename << endl;
		cout << "  Number of images = " << numberOfImages << endl;
		cout << "  SOM dimension = " << som_dim << "x" << som_dim << endl;
		cout << "  Number of iterations = " << numIter << endl;
		cout << "  Neuron dimension = " << neuron_dim << "x" << neuron_dim << endl;
		cout << "  Progress = " << progressFactor << endl;
		cout << "  Result file = " << resultFilename << endl;
		cout << "  Layout = " << layout << endl;
		cout << "  Initialization type = " << init << endl;
		cout << "  Seed = " << seed << endl;
		cout << "  Number of rotations = " << numberOfRotations << endl;
		cout << "  Number of CPU threads = " << numberOfThreads << endl;
		cout << "  Use CUDA = " << useCuda << endl;
	}
}

void InputData::print_usage() const
{
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
	        "    --verbose, -v           Print more output (yes, no, default = yes).\n"
	        "    --som-dimension         Dimension for quadratic SOM matrix (default = 10).\n"
	        "    --neuron-dimension, -d  Dimension for quadratic SOM neurons (default = image-size * sqrt(2)/2).\n"
	        "    --num-iter              Number of iterations (default = 1).\n"
	        "    --layout, -l            Layout of SOM (quadratic, hexagonal, default = quadratic).\n"
	        "    --seed, -s              Seed for random number generator (default = 1234).\n"
			"    --progress, -p          Print level of progress (default = 10%).\n"
	        "    --numrot, -n            Number of rotations (default = 360).\n"
	        "    --flip                  Switch off usage of mirrored images (yes, no, default = yes).\n"
	        "    --numthreads, -t        Number of CPU threads (default = auto).\n"
	        "    --init, -x              Type of SOM initialization (random, zero, default = zero).\n"
            "    --cuda                  Switch off CUDA acceleration (yes, no, default = yes).\n" << endl;
}

void stringToUpper(char* s)
{
	for (char *ps = s; *ps != '\0'; ++ps) *ps = toupper(*ps);
}
