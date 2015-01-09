/**
 * @file   SelfOrganizingMapLib/trainSelfOrganizingMap.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMap.h"
#include "SOM.h"
#include "UtilitiesLib/CheckArrays.h"
#include "UtilitiesLib/Error.h"
#include "UtilitiesLib/Filler.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/TimeAccumulator.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;
using namespace PINK;
using namespace chrono;

void SOM::training()
{
    cout << "  Starting C version of training.\n" << endl;

	// Memory allocation
	int rotatedImagesSize = inputData_.numberOfChannels * inputData_.numberOfRotations * inputData_.neuron_size;
	if (inputData_.useFlip) rotatedImagesSize *= 2;
	if (inputData_.verbose) cout << "  Size of rotated images = " << rotatedImagesSize * sizeof(float) << " bytes" << endl;
	vector<float> rotatedImages(rotatedImagesSize);

	if (inputData_.verbose) cout << "  Size of euclidean distance matrix = " << inputData_.som_size * sizeof(float) << " bytes" << endl;
	vector<float> euclideanDistanceMatrix(inputData_.som_size);

	if (inputData_.verbose) cout << "  Size of best rotation matrix = " << inputData_.som_size * sizeof(int) << " bytes" << endl;
	vector<int> bestRotationMatrix(inputData_.som_size);

	if (inputData_.verbose) cout << "  Size of SOM = " << getSizeInBytes() << " bytes\n" << endl;

	float progress = 0.0;
	float progressStep = 1.0 / inputData_.numIter / inputData_.numberOfImages;
	float nextProgressPrint = inputData_.progressFactor;
    int progressPrecision = rint(log10(1.0 / inputData_.progressFactor)) - 2;
    if (progressPrecision < 0) progressPrecision = 0;

	// Start timer
	auto startTime = steady_clock::now();
	const int maxTimer = 3;
	std::chrono::high_resolution_clock::duration timer[maxTimer] = {std::chrono::high_resolution_clock::duration::zero()};

	int interStoreCount = 0;
	int updateCount = 0;

	for (int iter = 0; iter != inputData_.numIter; ++iter)
	{
		for (ImageIterator<float> iterImage(inputData_.imagesFilename), iterEnd; iterImage != iterEnd; ++iterImage, ++updateCount)
		{
            if ((inputData_.progressFactor < 1.0 and progress > nextProgressPrint) or
                (inputData_.progressFactor >= 1.0 and updateCount != 0 and !(updateCount % static_cast<int>(inputData_.progressFactor))))
            {
                cout << "  Progress: " << setw(12) << updateCount << " updates, "
                     << fixed << setprecision(progressPrecision) << setw(3) << progress*100 << " % ("
                     << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;
                if (inputData_.verbose) {
                    cout << "  Time for image rotations = " << duration_cast<milliseconds>(timer[0]).count() << " ms" << endl;
                    cout << "  Time for euclidean distance = " << duration_cast<milliseconds>(timer[1]).count() << " ms" << endl;
                    cout << "  Time for SOM update = " << duration_cast<milliseconds>(timer[2]).count() << " ms" << endl;
                }

                if (inputData_.intermediate_storage != OFF) {
                    string interStoreFilename = inputData_.resultFilename;
                    if (inputData_.intermediate_storage == KEEP) {
                        interStoreFilename.insert(interStoreFilename.find_last_of("."), "_" + to_string(interStoreCount));
                        ++interStoreCount;
                    }
                    if (inputData_.verbose) cout << "  Write intermediate SOM to " << interStoreFilename << " ... " << flush;
                    write(interStoreFilename);
                    if (inputData_.verbose) cout << "done." << endl;
                }

                nextProgressPrint += inputData_.progressFactor;
                startTime = steady_clock::now();
                for (int i(0); i < maxTimer; ++i) timer[i] = std::chrono::high_resolution_clock::duration::zero();
            }
            progress += progressStep;

            {
                TimeAccumulator localTimeAccumulator(timer[0]);
                generateRotatedImages(&rotatedImages[0], iterImage->getPointerOfFirstPixel(), inputData_.numberOfRotations,
                    inputData_.image_dim, inputData_.neuron_dim, inputData_.useFlip, inputData_.interpolation,
                    inputData_.numberOfChannels);
            }

            {
                TimeAccumulator localTimeAccumulator(timer[1]);
                generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
                    inputData_.som_size, &som_[0], inputData_.numberOfChannels * inputData_.neuron_size,
                    inputData_.numberOfRotationsAndFlip, &rotatedImages[0]);
            }

            {
                TimeAccumulator localTimeAccumulator(timer[2]);
                int bestMatch = findBestMatchingNeuron(&euclideanDistanceMatrix[0], inputData_.som_size);
                updateCounter(bestMatch);
                updateNeurons(&rotatedImages[0], bestMatch, &bestRotationMatrix[0]);
            }
		}
	}

	cout << "  Progress: " << setw(12) << updateCount << " updates, 100 % ("
		 << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;
    if (inputData_.verbose) {
        cout << "  Time for image rotations = " << duration_cast<milliseconds>(timer[0]).count() << " ms" << endl;
        cout << "  Time for euclidean distance = " << duration_cast<milliseconds>(timer[1]).count() << " ms" << endl;
        cout << "  Time for SOM update = " << duration_cast<milliseconds>(timer[2]).count() << " ms" << endl;
    }

	if (inputData_.verbose) cout << "  Write final SOM to " << inputData_.resultFilename << " ... " << flush;
	write(inputData_.resultFilename);
	if (inputData_.verbose) cout << "done." << endl;

    printUpdateCounter();
}
