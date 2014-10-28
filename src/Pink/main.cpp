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
    #include "CudaLib/cuda_print_properties.h"
#endif

using namespace std;
using namespace PINK;

void print_usage()
{
	cout << endl;
	cout << "  USAGE: Pink -i <path for image-file> -r <path for result-file>\n" << endl;
	cout << "  Non-optional options:\n" << endl;
	cout << "    --images, -i         File " << endl;
	cout << "    --result, -r         File for final SOM matrix.\n" << endl;
	cout << "  Optional options:\n" << endl;
	cout << "    --verbose, -v        Print more output." << endl;
	cout << "    --dimension, -d      Dimension for quadratic SOM matrix (default = 10)." << endl;
	cout << "    --layout, -l         Layout of SOM (quadratic, hexagonal, default = quadratic)." << endl;
	cout << "    --seed, -s           Seed for random number generator (default = 1234)." << endl;
	cout << "    --numrot, -n         Number of rotations (default = 360)." << endl;
	cout << "    --numthreads, -t     Number of CPU threads (default = auto)." << endl;
}

int main (int argc, char **argv)
{
	int c;
	int digit_optind = 0;
	int verbose = 0;
	char *imagesFilename = 0;
	int som_dim = 10;
	Layout layout = QUADRATIC;
	char *resultFilename = 0;
	int seed = 1234;
	int numberOfRotations = 360;
	int numberOfThreads = -1;

	static struct option long_options[] = {
		{"verbose",    0, 0, 'v'},
		{"images",     1, 0, 'i'},
		{"dimension",  1, 0, 'd'},
		{"result",     1, 0, 'r'},
		{"layout",     1, 0, 'l'},
		{"seed",       1, 0, 's'},
		{"numrot",     1, 0, 'n'},
		{"numthreads", 1, 0, 't'},
		{NULL, 0, NULL, 0}
	};
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "vi:d:r:l:s:n:t:", long_options, &option_index)) != -1)
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
		case '?':
			break;
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

	if (!imagesFilename or !resultFilename) {
		print_usage();
		return 1;
	}

	if (numberOfThreads == -1) numberOfThreads = omp_get_num_procs();
	else omp_set_num_threads(numberOfThreads);

	if (verbose) {
		cout << "  images = " << imagesFilename << endl;
		cout << "  dimension = " << som_dim << endl;
		cout << "  result = " << resultFilename << endl;
		cout << "  layout = " << layout << endl;
		cout << "  seed = " << seed << endl;
		cout << "  number of rotations = " << numberOfRotations << endl;
		cout << "  Number of CPU threads = " << numberOfThreads << endl;
	}

    #if PINK_USE_CUDA
	    if (verbose) cuda_print_properties();
    #endif

	ImageIterator<float> iterImage(imagesFilename);
	if (verbose) cout << "Image dimension = " << iterImage->getWidth() << "x" << iterImage->getHeight() << endl;

	if (iterImage->getWidth() != iterImage->getHeight()) {
		cout << "Only quadratic images are supported.";
		return 1;
	}

	int numberOfImages = iterImage.number();
	if (verbose) cout << "number of images = " << numberOfImages << endl;
	int image_dim = iterImage->getWidth();
	int image_size = iterImage->getWidth() * iterImage->getHeight();
    int som_size = som_dim * som_dim;

	// Initialize SOM
	float *som = (float *)malloc(som_size * image_size * sizeof(float));
	fillRandom(som, som_size * image_size, seed);

	float progress = 0.0;
	float progressStep = 1.0 / numberOfImages;
	float nextProgressPrint = 0.0;
	for (int i = 0; iterImage != ImageIterator<float>(); ++i, ++iterImage)
	{
	    if (verbose) {
			if (progress >= nextProgressPrint) {
				cout << "Progress: " << fixed << setprecision(0) << progress*100 << " %" << endl;
				nextProgressPrint += 0.1;
			}
		    progress += progressStep;
	    }

//		stringstream ss;
//		ss << "image" << i << ".bin";
//		iterImage->writeBinary(ss.str());
//		iterImage->show();

		float *image = iterImage->getPointerOfFirstPixel();
		int image_dim = iterImage->getWidth();
		int image_size = iterImage->getWidth() * iterImage->getHeight();

		float *rotatedImages = (float *)malloc(2 * numberOfRotations * image_size * sizeof(float));
		generateRotatedImages(rotatedImages, image, numberOfRotations, image_dim);

//		stringstream ss2;
//		ss2 << "rotatedImage" << i << ".bin";
//		writeRotatedImages(rotatedImages, image_dim, numberOfRotations, ss2.str());
//		showRotatedImages(rotatedImages, image_dim, numberOfRotations);

		float *similarityMatrix = (float *)malloc(som_size * sizeof(float));
		int *bestRotationMatrix = (int *)malloc(som_size * sizeof(int));
		generateEuclideanDistanceMatrix(similarityMatrix, bestRotationMatrix, som_dim, som, image_dim, numberOfRotations, rotatedImages);

		Point bestMatch = findBestMatchingNeuron(similarityMatrix, som_dim);

		//cout << "bestMatch = " << bestMatch << endl;

		updateNeurons(som_dim, som, image_dim, rotatedImages, bestMatch, bestRotationMatrix);

//		stringstream ss3;
//		ss3 << "som" << i << ".bin";
//		writeSOM(som, som_dim, image_dim, ss3.str());
//		showSOM(som, som_dim, image_dim);

		free(rotatedImages);
		free(similarityMatrix);
		free(bestRotationMatrix);
	}

    if (verbose) {
	    cout << "Progress: 100 %\n" << endl;
	    cout << "Write final SOM to " << resultFilename << " ..." << endl;
    }

	writeSOM(som, som_dim, image_dim, resultFilename);
	free(som);

    if (verbose) cout << "\nAll done.\n" << endl;
	return 0;
}
