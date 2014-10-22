/**
 * @file   main.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib/cuda_print_properties.h"
#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include <unistd.h>
#include <iostream>

int main (int argc, char **argv)
{
	std::string inputFilename, outputFilename;
	int index;
	int c;

	opterr = 0;

	while ((c = getopt (argc, argv, "i:o:")) != -1)
	switch (c)
	{
		case 'i':
			inputFilename = optarg;
			break;
		case 'o':
			outputFilename = optarg;
			break;
		case '?':
			if (optopt == 'c')
			    std::cerr << "Option -" << optopt << "requires an argument." << std::endl;
			else if (isprint (optopt))
			    std::cerr << "Unknown option -" << optopt << "." << std::endl;
			else
			    std::cerr << "Unknown option character -" << optopt << "." << std::endl;
			return 1;
		default:
			abort();
	}

	std::cout << "inputFilename = " << inputFilename << std::endl;
	std::cout << "outputFilename = " << outputFilename << std::endl;

    cuda_print_properties();

	for (PINK::ImageIterator<float> iterCur(inputFilename), iterEnd; iterCur != iterEnd; ++iterCur)
	{
		float *image = &(*iterCur)->getPixel()[0];
	}

	std::cout << "All done." << std::endl;
	return 0;
}
