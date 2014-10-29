/**
 * @file   main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include <cmath>
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#if PINK_USE_CUDA
    #include "CudaLib/CudaLib.h"
#endif

using namespace std;
using namespace PINK;
using namespace chrono;

void print_header()
{
	cout << "\n"
	        "  ************************************************************************\n"
	        "  *   Parallel orientation Invariant Non-parametric Kohonen-map (PINK)   *\n"
	        "  ************************************************************************\n" << endl;
}

void print_usage()
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
	        "    --verbose, -v           Print more output (default = off).\n"
	        "    --som-dimension         Dimension for quadratic SOM matrix (default = 10).\n"
	        "    --neuron-dimension, -d  Dimension for quadratic SOM neurons (default = image-size * sqrt(2)/2).\n"
	        "    --num-iter              Number of iterations (default = 1).\n"
	        "    --layout, -l            Layout of SOM (quadratic, hexagonal, default = quadratic).\n"
	        "    --seed, -s              Seed for random number generator (default = 1234).\n"
			"    --progress, -p          Print level of progress (default = 10%).\n"
	        "    --numrot, -n            Number of rotations (default = 360).\n"
	        "    --numthreads, -t        Number of CPU threads (default = auto).\n"
	        "    --init, -x              Type of SOM initialization (random, zero, default = zero).\n" << endl;
}

int main (int argc, char **argv)
{
	// Start timer
	const auto startTime = steady_clock::now();

	int c;
	int digit_optind = 0;
	int verbose = 0;
	char *imagesFilename = 0;
	int som_dim = 10;
	int neuron_dim = -1;
	Layout layout = QUADRATIC;
	char *resultFilename = 0;
	int seed = 1234;
	int numberOfRotations = 360;
	int numberOfThreads = -1;
	SOMInitialization init = ZERO;
	bool useCuda = false;
	int numIter = 1;
	float progressFactor = 0.1;

	print_header();

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
		{NULL, 0, NULL, 0}
	};
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "vi:d:r:l:s:n:t:x:p:", long_options, &option_index)) != -1)
	{
		int this_option_optind = optind ? optind : 1;
		switch (c) {
		case 'v':
			verbose = 1;
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
			if (strcmp(optarg, "QUADRATIC") != 0) layout = QUADRATIC;
			else if (strcmp(optarg, "HEXAGONAL") != 0) layout = HEXAGONAL;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				return 1;
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
				return 1;
			}
			break;
		case 't':
			numberOfThreads = atoi(optarg);
			break;
		case 'x':
			stringToUpper(optarg);
			if (strcmp(optarg, "ZERO") != 0) init = ZERO;
			else if (strcmp(optarg, "RANDOM") != 0) init = RANDOM;
			else {
				printf ("optarg = %s\n", optarg);
				printf ("Unkown option %o\n", c);
				print_usage();
				return 1;
			}
			break;
		case '?':
			printf ("Unkown option %o\n", c);
			print_usage();
			return 1;
		default:
			printf ("Unkown option %o\n", c);
			print_usage();
			return 1;
		}
	}

	if (optind < argc) {
		cout << "Unkown argv elements: ";
		while (optind < argc) cout << argv[optind++] << " ";
		cout << endl;
		print_usage();
		return 1;
	}

	// Check if all non-optional arguments are set
	if (!imagesFilename or !resultFilename) {
		cout << "Missing non-optional argument." << endl;
		print_usage();
		return 1;
	}

	if (numberOfThreads == -1) numberOfThreads = omp_get_num_procs();
	else omp_set_num_threads(numberOfThreads);

	ImageIterator<float> iterImage(imagesFilename);
	if (verbose) cout << "  Image dimension = " << iterImage->getWidth() << "x" << iterImage->getHeight() << endl;

	if (iterImage->getWidth() != iterImage->getHeight()) {
		cout << "Only quadratic images are supported.";
		return 1;
	}

	int numberOfImages = iterImage.number();
	int image_dim = iterImage->getWidth();
	int image_size = iterImage->getWidth() * iterImage->getHeight();
    int som_size = som_dim * som_dim;

    if (neuron_dim == -1) neuron_dim = image_dim * sqrt(2.0) / 2.0;
    if (neuron_dim > image_dim) {
		cout << "Neuron dimension must be smaller or equal to image dimension.";
		return 1;
    }
    if ((image_dim - neuron_dim)%2) --neuron_dim;
    int neuron_size = neuron_dim * neuron_dim;

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
	}

    #if PINK_USE_CUDA
	    if (useCuda and verbose) cuda_print_properties();
    #endif

	// Memory allocation
	if (verbose) cout << "\n  Size of SOM = " << som_size * neuron_size * sizeof(float) << " bytes" << endl;
	float *som = (float *)malloc(som_size * neuron_size * sizeof(float));

	if (verbose) cout << "  Size of rotated images = " << 2 * numberOfRotations * neuron_size * sizeof(float) << " bytes" << endl;
	float *rotatedImages = (float *)malloc(2 * numberOfRotations * neuron_size * sizeof(float));

	if (verbose) cout << "  Size of euclidean distance matrix = " << som_size * sizeof(float) << " bytes" << endl;
	float *euclideanDistanceMatrix = (float *)malloc(som_size * sizeof(float));

	if (verbose) cout << "  Size of best rotation matrix = " << som_size * sizeof(int) << " bytes" << endl;
	int *bestRotationMatrix = (int *)malloc(som_size * sizeof(int));

	// Initialize SOM
	if (init == RANDOM) fillRandom(som, som_size * neuron_size, seed);
	else if (init == ZERO) fillZero(som, som_size * neuron_size);

    if (verbose) {
    	cout << "\n  Write initial SOM to initial_som.bin ...\n" << endl;
    	writeSOM(som, som_dim, neuron_dim, "initial_som.bin");
    }

	float progress = 0.0;
	float progressStep = 1.0 / numIter / numberOfImages;
	float nextProgressPrint = 0.0;

	for (int iter = 0; iter != numIter; ++iter)
	{
		int i = 0;
		for (ImageIterator<float> iterImage(imagesFilename),iterEnd; iterImage != iterEnd; ++i, ++iterImage)
		{
			if (verbose) {
				if (progress >= nextProgressPrint) {
					cout << "  Progress: " << fixed << setprecision(0) << progress*100 << " %" << endl;
					nextProgressPrint += progressFactor;
				}
				progress += progressStep;
			}

	//		stringstream ss;
	//		ss << "image" << i << ".bin";
	//		iterImage->writeBinary(ss.str());

			float *image = iterImage->getPointerOfFirstPixel();
			generateRotatedImages(rotatedImages, image, numberOfRotations, image_dim, neuron_dim);

	//		stringstream ss2;
	//		ss2 << "rotatedImage" << i << ".bin";
	//		writeRotatedImages(rotatedImages, image_dim, numberOfRotations, ss2.str());

			generateEuclideanDistanceMatrix(euclideanDistanceMatrix, bestRotationMatrix, som_dim, som, neuron_dim, numberOfRotations, rotatedImages);

			Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, som_dim);

			//cout << "bestMatch = " << bestMatch << endl;

			updateNeurons(som_dim, som, neuron_dim, rotatedImages, bestMatch, bestRotationMatrix);

	//		stringstream ss3;
	//		ss3 << "som" << i << ".bin";
	//		writeSOM(som, som_dim, image_dim, ss3.str());
		}
	}

	free(rotatedImages);
	free(euclideanDistanceMatrix);
	free(bestRotationMatrix);

    if (verbose) {
	    cout << "  Progress: 100 %\n" << endl;
	    cout << "  Write final SOM to " << resultFilename << " ..." << endl;
    }

	writeSOM(som, som_dim, neuron_dim, resultFilename);
	free(som);

	// Stop and print timer
	const auto stopTime = steady_clock::now();
	const auto duration = stopTime - startTime;
	cout << "\n  Total time (hh:mm:ss): "
		 << setfill('0') << setw(2) << duration_cast<hours>(duration).count() << ":"
		 << setfill('0') << setw(2) << duration_cast<minutes>(duration % hours(1)).count() << ":"
		 << setfill('0') << setw(2) << duration_cast<seconds>(duration % minutes(1)).count() << endl;

    cout << "\n  All done.\n" << endl;
	return 0;
}
