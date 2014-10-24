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
#include <stdlib.h>

#if PINK_USE_CUDA
    #include "CudaLib/cuda_print_properties.h"
#endif

using namespace std;
using namespace PINK;

void print_usage()
{
	cout << "  Non-optional options:\n" << endl;
	cout << "    --images, -i         File " << endl;
	cout << "    --result, -r         File for final SOM matrix.\n" << endl;
	cout << "  Optional options:\n" << endl;
	cout << "    --verbose, -v        Print more output." << endl;
	cout << "    --dimension, -d      Dimension for quadratic SOM matrix (default = 10)." << endl;
	cout << "    --layout, -l         Layout of SOM (quadratic, hexagonal, default = quadratic)." << endl;
	cout << "    --seed, -s           Seed for random number generator (default = 1234)." << endl;
	cout << "    --numrot, -n         Number of rotations (default = 360)." << endl;
}

int main (int argc, char **argv)
{
	int c;
	int digit_optind = 0;
	int verbose = 0;
	char *imagesFilename = 0;
	int som_dim = 10;
	char *resultFilename = 0;
	int seed = 1234;
	int numberOfRotations = 360;
	Layout layout = QUADRATIC;

	static struct option long_options[] = {
		{"verbose",   0, 0, 'v'},
		{"images",    1, 0, 'i'},
		{"dimension", 1, 0, 'd'},
		{"result",    1, 0, 'r'},
		{"layout",    1, 0, 'l'},
		{"seed",      1, 0, 's'},
		{"numrot",    1, 0, 'n'},
		{NULL, 0, NULL, 0}
	};
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "vi:d:r:l:s:n:", long_options, &option_index)) != -1)
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

	cout << "verbose = " << verbose << endl;
	cout << "images = " << imagesFilename << endl;
	cout << "dimension = " << som_dim << endl;
	cout << "result = " << resultFilename << endl;
	cout << "layout = " << layout << endl;
	cout << "seed = " << seed << endl;
	cout << "numberOfRotations = " << numberOfRotations << endl;

    #if PINK_USE_CUDA
	    if (verbose) cuda_print_properties();
    #endif

	ImageIterator<float> iterImage(imagesFilename);
	if (verbose) cout << "Image dimension = " << iterImage->getWidth() << "x" << iterImage->getHeight() << endl;

	if (iterImage->getWidth() != iterImage->getHeight()) {
		cout << "Only quadratic images are supported.";
		return 1;
	}

	int image_dim = iterImage->getWidth();
	int image_size = iterImage->getWidth() * iterImage->getHeight();
    int som_size = som_dim * som_dim;

	// Initialize SOM
	float *som = (float *)malloc(som_size * image_size * sizeof(float));
	fillRandom(som, som_size * image_size, seed);

	for (; iterImage != ImageIterator<float>(); ++iterImage)
	{
		float *image = iterImage->getPointerOfFirstPixel();
		int image_dim = iterImage->getWidth();
		int image_size = iterImage->getWidth() * iterImage->getHeight();

		float *rotatedImages = (float *)malloc(2 * numberOfRotations * image_size * sizeof(float));
		generateRotatedImages(rotatedImages, image, numberOfRotations, image_dim);

		float *similarityMatrix = (float *)malloc(som_size * sizeof(float));
		int *bestRotationMatrix = (int *)malloc(som_size * sizeof(int));
		generateSimilarityMatrix(similarityMatrix, bestRotationMatrix, som_dim, som, image_dim, numberOfRotations, rotatedImages);

		Point bestMatch = findBestMatchingNeuron(similarityMatrix, som_dim);

		cout << "bestMatch = " << bestMatch << endl;

		updateNeurons(som_dim, som, image_dim, image, bestMatch);

		showSOM(som, som_dim, image_dim);
	}

    if (verbose) cout << "\nAll done.\n" << endl;
	return 0;
}
