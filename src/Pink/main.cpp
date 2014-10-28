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
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <omp.h>

#if PINK_USE_CUDA
    #include "CudaLib/CudaLib.h"
#endif

using namespace std;
using namespace PINK;

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
	        "  USAGE: Pink -i <path for image-file> -r <path for result-file>\n"
			"\n"
	        "  Non-optional options:\n"
			"\n"
	        "    --image-file, -i        File with images.\n"
	        "    --result-file, -r       File for final SOM matrix.\n"
	        "    --image-dimension, -d   Dimension for quadratic SOM matrix (default = 10).\n"
			"\n"
	        "  Optional options:\n"
			"\n"
	        "    --verbose, -v           Print more output (default = off).\n"
	        "    --som-dimension         Dimension for quadratic SOM matrix (default = 10).\n"
	        "    --layout, -l            Layout of SOM (quadratic, hexagonal, default = quadratic).\n"
	        "    --seed, -s              Seed for random number generator (default = 1234).\n"
	        "    --numrot, -n            Number of rotations (default = 360).\n"
	        "    --numthreads, -t        Number of CPU threads (default = auto).\n"
	        "    --init, -x              Type of SOM initialization (random, zero, default = zero).\n" << endl;
}

int main (int argc, char **argv)
{
	int c;
	int digit_optind = 0;
	int verbose = 0;
	char *imagesFilename = 0;
	int som_dim = 10;
	int som_image_dim = -1;
	Layout layout = QUADRATIC;
	char *resultFilename = 0;
	int seed = 1234;
	int numberOfRotations = 360;
	int numberOfThreads = -1;
	SOMInitialization init = ZERO;

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
		{NULL, 0, NULL, 0}
	};
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "vi:d:r:l:s:n:t:x:", long_options, &option_index)) != -1)
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
			som_image_dim = atoi(optarg);
			break;
		case 0:
			som_dim = atoi(optarg);
			break;
		case 'r':
			resultFilename = optarg;
			break;
		case 'l':
			if (stringToUpper(optarg) == "QUADRATIC") layout = QUADRATIC;
			else if (optarg == "HEXAGONAL") layout = HEXAGONAL;
			else {
				printf ("Unkown option %o\n", c);
				print_usage();
				return 1;
			}
			break;
		case 's':
			seed = atoi(optarg);
			break;
		case 'n':
			numberOfRotations = atoi(optarg);
			if (numberOfRotations < 1 or numberOfRotations > 360) {
				printf ("Number of rotations must be between 1 and 360.\n\n");
				print_usage();
				return 1;
			}
			break;
		case 't':
			numberOfThreads = atoi(optarg);
			break;
		case 'x':
			if (stringToUpper(optarg) == "ZERO") init = ZERO;
			else if (optarg == "RANDOM") init = RANDOM;
			else {
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
	if (!imagesFilename or !resultFilename or som_image_dim == -1) {
		cout << "Missing non-optional argument." << endl;
		print_usage();
		return 1;
	}

	if (numberOfThreads == -1) numberOfThreads = omp_get_num_procs();
	else omp_set_num_threads(numberOfThreads);

	if (verbose) {
		cout << "  Image file = " << imagesFilename << endl;
		cout << "  SOM dimension = " << som_dim << "x" << som_dim << endl;
		cout << "  SOM image dimension = " << som_image_dim << "x" << som_image_dim << endl;
		cout << "  Result file = " << resultFilename << endl;
		cout << "  Layout = " << layout << endl;
		cout << "  Initialization type = " << init << endl;
		cout << "  Seed = " << seed << endl;
		cout << "  Number of rotations = " << numberOfRotations << endl;
		cout << "  Number of CPU threads = " << numberOfThreads << endl;
	}

    #if PINK_USE_CUDA
	    if (verbose) cuda_print_properties();
    #endif

	ImageIterator<float> iterImage(imagesFilename);
	if (verbose) cout << "  Image dimension = " << iterImage->getWidth() << "x" << iterImage->getHeight() << endl;

	if (iterImage->getWidth() != iterImage->getHeight()) {
		cout << "Only quadratic images are supported.";
		return 1;
	}

	int numberOfImages = iterImage.number();
	if (verbose) cout << "  Number of images = " << numberOfImages << endl;
	int image_dim = iterImage->getWidth();
	int image_size = iterImage->getWidth() * iterImage->getHeight();
    int som_size = som_dim * som_dim;

	// Initialize SOM
	if (verbose) cout << "  Size of SOM = " << som_size * image_size * sizeof(float) << " bytes" << endl;
	float *som = (float *)malloc(som_size * image_size * sizeof(float));

	if (init == RANDOM) fillRandom(som, som_size * image_size, seed);
	else if (init == ZERO) fillZero(som, som_size * image_size);

    if (verbose) cout << "  Write initial SOM to initial_som.bin ..." << endl;
	writeSOM(som, som_dim, image_dim, "initial_som.bin");

	if (verbose) cout << "  Size of rotated images = " << 2 * numberOfRotations * image_size * sizeof(float) << " bytes" << endl;
	float *rotatedImages = (float *)malloc(2 * numberOfRotations * image_size * sizeof(float));

	if (verbose) cout << "  Size of euclidean distance matrix = " << 2 * numberOfRotations * image_size * sizeof(float) << " bytes" << endl;
	float *euclideanDistanceMatrix = (float *)malloc(som_size * sizeof(float));

	if (verbose) cout << "  Size of best rotation matrix = " << som_size * sizeof(int) << " bytes" << endl;
	int *bestRotationMatrix = (int *)malloc(som_size * sizeof(int));

	float progress = 0.0;
	float progressStep = 1.0 / numberOfImages;
	float nextProgressPrint = 0.0;

	cout << endl;

	for (int i = 0; iterImage != ImageIterator<float>(); ++i, ++iterImage)
	{
	    if (verbose) {
			if (progress >= nextProgressPrint) {
				cout << "  Progress: " << fixed << setprecision(0) << progress*100 << " %" << endl;
				nextProgressPrint += 0.1;
			}
		    progress += progressStep;
	    }

//		stringstream ss;
//		ss << "image" << i << ".bin";
//		iterImage->writeBinary(ss.str());
//		iterImage->show();

		float *image = iterImage->getPointerOfFirstPixel();
		generateRotatedImages(rotatedImages, image, numberOfRotations, image_dim);

//		stringstream ss2;
//		ss2 << "rotatedImage" << i << ".bin";
//		writeRotatedImages(rotatedImages, image_dim, numberOfRotations, ss2.str());
//		showRotatedImages(rotatedImages, image_dim, numberOfRotations);

		generateEuclideanDistanceMatrix(euclideanDistanceMatrix, bestRotationMatrix, som_dim, som, image_dim, numberOfRotations, rotatedImages);

		Point bestMatch = findBestMatchingNeuron(euclideanDistanceMatrix, som_dim);

		if (verbose >= 2) cout << "bestMatch = " << bestMatch << endl;

		updateNeurons(som_dim, som, image_dim, rotatedImages, bestMatch, bestRotationMatrix);

//		stringstream ss3;
//		ss3 << "som" << i << ".bin";
//		writeSOM(som, som_dim, image_dim, ss3.str());
//		showSOM(som, som_dim, image_dim);
	}

	free(rotatedImages);
	free(euclideanDistanceMatrix);
	free(bestRotationMatrix);

    if (verbose) {
	    cout << "  Progress: 100 %\n" << endl;
	    cout << "  Write final SOM to " << resultFilename << " ..." << endl;
    }

	writeSOM(som, som_dim, image_dim, resultFilename);
	free(som);

    if (verbose) cout << "\n  All done.\n" << endl;
	return 0;
}
