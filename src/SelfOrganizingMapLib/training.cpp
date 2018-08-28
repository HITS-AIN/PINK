/**
 * @file   SelfOrganizingMapLib/trainSelfOrganizingMap.cpp
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <iostream>
#include <iomanip>

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

namespace pink {

void SOM::training()
{
    std::cout << "  Starting C version of training.\n" << std::endl;

    // Memory allocation
    int rotatedImagesSize = inputData_.numberOfChannels * inputData_.numberOfRotations * inputData_.neuron_size;
    if (inputData_.useFlip) rotatedImagesSize *= 2;
    if (inputData_.verbose) std::cout << "  Size of rotated images = " << rotatedImagesSize * sizeof(float) << " bytes" << std::endl;
    std::vector<float> rotatedImages(rotatedImagesSize);

    if (inputData_.verbose) std::cout << "  Size of euclidean distance matrix = " << inputData_.som_size * sizeof(float) << " bytes" << std::endl;
    std::vector<float> euclideanDistanceMatrix(inputData_.som_size);

    if (inputData_.verbose) std::cout << "  Size of best rotation matrix = " << inputData_.som_size * sizeof(int) << " bytes" << std::endl;
    std::vector<int> bestRotationMatrix(inputData_.som_size);

    if (inputData_.verbose) std::cout << "  Size of SOM = " << getSizeInBytes() << " bytes\n" << std::endl;

    float progress = 0.0;
    float progressStep = 1.0 / inputData_.numIter / inputData_.numberOfImages;
    float nextProgressPrint = inputData_.progressFactor;
    int progressPrecision = rint(log10(1.0 / inputData_.progressFactor)) - 2;
    if (progressPrecision < 0) progressPrecision = 0;

    // Start timer
    auto startTime = myclock::now();
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
                std::cout << "  Progress: " << std::setw(12) << updateCount << " updates, "
                     << std::fixed << std::setprecision(progressPrecision) << std::setw(3) << progress*100 << " % ("
                     << std::chrono::duration_cast<std::chrono::seconds>(myclock::now() - startTime).count() << " s)" << std::endl;
                if (inputData_.verbose) {
                    std::cout << "  Time for image rotations = " << std::chrono::duration_cast<std::chrono::milliseconds>(timer[0]).count() << " ms" << std::endl;
                    std::cout << "  Time for euclidean distance = " << std::chrono::duration_cast<std::chrono::milliseconds>(timer[1]).count() << " ms" << std::endl;
                    std::cout << "  Time for SOM update = " << std::chrono::duration_cast<std::chrono::milliseconds>(timer[2]).count() << " ms" << std::endl;
                }

                if (inputData_.intermediate_storage != OFF) {
                    std::string interStoreFilename = inputData_.resultFilename;
                    if (inputData_.intermediate_storage == KEEP) {
                        interStoreFilename.insert(interStoreFilename.find_last_of("."), "_" + std::to_string(interStoreCount));
                        ++interStoreCount;
                    }
                    if (inputData_.verbose) std::cout << "  Write intermediate SOM to " << interStoreFilename << " ... " << std::flush;
                    write(interStoreFilename);
                    if (inputData_.verbose) std::cout << "done." << std::endl;
                }

                nextProgressPrint += inputData_.progressFactor;
                startTime = myclock::now();
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

    std::cout << "  Progress: " << std::setw(12) << updateCount << " updates, 100 % ("
         << std::chrono::duration_cast<std::chrono::seconds>(myclock::now() - startTime).count() << " s)" << std::endl;
    if (inputData_.verbose) {
        std::cout << "  Time for image rotations = " << std::chrono::duration_cast<std::chrono::milliseconds>(timer[0]).count() << " ms" << std::endl;
        std::cout << "  Time for euclidean distance = " << std::chrono::duration_cast<std::chrono::milliseconds>(timer[1]).count() << " ms" << std::endl;
        std::cout << "  Time for SOM update = " << std::chrono::duration_cast<std::chrono::milliseconds>(timer[2]).count() << " ms" << std::endl;
    }

    if (inputData_.verbose) std::cout << "  Write final SOM to " << inputData_.resultFilename << " ... " << std::flush;
    write(inputData_.resultFilename);
    if (inputData_.verbose) std::cout << "done." << std::endl;

    printUpdateCounter();
}

} // namespace pink
