/**
 * @file   main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
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
	cout << "  Options:\n" << endl;
	cout << "    --verbose, -v        Print more output." << endl;
	cout << "    --images, -i         File " << endl;
	cout << "    --dimension, -d      Dimension for quadratic SOM matrix (default = 10)." << endl;
	cout << "    --result, -r         File for final SOM matrix." << endl;
}

int main (int argc, char **argv)
{
	int c;
	int digit_optind = 0;
	int verbose = 0;
	char *imagesFilename = 0;
	int som_dim = 10;
	char *resultFilename = 0;
	static struct option long_options[] = {
		{"verbose",   0, 0, 'v'},
		{"images",    1, 0, 'i'},
		{"dimension", 1, 0, 'd'},
		{"result",    1, 0, 'r'},
		{NULL, 0, NULL, 0}
	};
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "vi:d:r:", long_options, &option_index)) != -1)
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
		case '?':
			break;
		default:
			printf ("Unkown option %o\n", c);
			print_usage();
		}
	}
	if (optind < argc) {
		printf ("non-option ARGV-elements: ");
		while (optind < argc)
			printf ("%s ", argv[optind++]);
		printf ("\n");
	}

	cout << "verbose = " << verbose << endl;
	cout << "images = " << imagesFilename << endl;
	cout << "dimension = " << som_dim << endl;
	cout << "result = " << resultFilename << endl;

    #if PINK_USE_CUDA
	    if (verbose) cuda_print_properties();
    #endif

    int som_size = som_dim * som_dim;

	ImageIterator<float> iterImage(imagesFilename);
	if (verbose) cout << "Image dimension = " << iterImage->getWidth() << "x" << iterImage->getHeight() << endl;

	int image_size = iterImage->getWidth() * iterImage->getHeight();
    float *som = (float *)malloc(som_size*image_size*sizeof(float));

    if (verbose) cout << "\nAll done.\n" << endl;
	return 0;
}
