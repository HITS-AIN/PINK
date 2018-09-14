/**
 * @file   SelfOrganizingMapLib/mapping.cpp
 * @date   Nov 28, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <iomanip>
#include <iostream>

#include "ImageProcessingLib/ImageIterator.h"
#include "SelfOrganizingMap.h"
#include "SOM.h"
#include "UtilitiesLib/Error.h"

namespace pink {

void SOM::mapping()
{
    std::cout << "  Starting C version of mapping.\n" << std::endl;

    // Open result file
    std::ofstream resultFile(inputData_.resultFilename);
    if (!resultFile) fatalError("Error opening " + inputData_.resultFilename);
    resultFile.write((char*)&inputData_.numberOfImages, sizeof(int));
    resultFile.write((char*)&inputData_.som_width, sizeof(int));
    resultFile.write((char*)&inputData_.som_height, sizeof(int));
    resultFile.write((char*)&inputData_.som_depth, sizeof(int));

    std::ofstream write_rot_flip_file;
    if (inputData_.write_rot_flip) {
        write_rot_flip_file.open(inputData_.rot_flip_filename);
        if (!write_rot_flip_file) fatalError("Error opening " + inputData_.rot_flip_filename);
        write_rot_flip_file.write((char*)&inputData_.numberOfImages, sizeof(int));
        write_rot_flip_file.write((char*)&inputData_.som_width, sizeof(int));
        write_rot_flip_file.write((char*)&inputData_.som_height, sizeof(int));
        write_rot_flip_file.write((char*)&inputData_.som_depth, sizeof(int));
    }

    // Memory allocation
    int rotatedImagesSize = inputData_.numberOfChannels * inputData_.numberOfRotations * inputData_.neuron_size;
    if (inputData_.useFlip) rotatedImagesSize *= 2;
    if (inputData_.verbose) std::cout << "  Size of rotated images = " << rotatedImagesSize * sizeof(float) << " bytes" << std::endl;
    std::vector<float> rotatedImages(rotatedImagesSize);

    if (inputData_.verbose) std::cout << "  Size of euclidean distance matrix = " << inputData_.som_size * sizeof(float) << " bytes" << std::endl;
    std::vector<float> euclideanDistanceMatrix(inputData_.som_size);

    if (inputData_.verbose) std::cout << "  Size of best rotation matrix = " << inputData_.som_size * sizeof(int) << " bytes\n" << std::endl;
    std::vector<int> bestRotationMatrix(inputData_.som_size);

    float angleStepRadians = 2.0 * M_PI / inputData_.numberOfRotations;

    float progress = 0.0;
    float progressStep = 1.0 / inputData_.numberOfImages;
    float nextProgressPrint = inputData_.progressFactor;
    int progressPrecision = rint(log10(1.0 / inputData_.progressFactor)) - 2;
    if (progressPrecision < 0) progressPrecision = 0;

    // Start timer
    auto startTime = myclock::now();
    int updateCount = 0;

    for (ImageIterator<float> iterImage(inputData_.imagesFilename), iterEnd; iterImage != iterEnd; ++iterImage, ++updateCount)
    {
        if ((inputData_.progressFactor < 1.0 and progress > nextProgressPrint) or
            (inputData_.progressFactor >= 1.0 and updateCount != 0 and !(updateCount % static_cast<int>(inputData_.progressFactor))))
        {
            std::cout << "  Progress: " << std::setw(12) << updateCount << " updates, "
                 << std::fixed << std::setprecision(progressPrecision) << std::setw(3) << progress*100 << " % ("
                 << std::chrono::duration_cast<std::chrono::seconds>(myclock::now() - startTime).count() << " s)" << std::endl;

            nextProgressPrint += inputData_.progressFactor;
            startTime = myclock::now();
        }
        progress += progressStep;

        generateRotatedImages(&rotatedImages[0], iterImage->getPointerOfFirstPixel(), inputData_.numberOfRotations,
            inputData_.image_dim, inputData_.neuron_dim, inputData_.useFlip, inputData_.interpolation,
            inputData_.numberOfChannels);

        generateEuclideanDistanceMatrix(&euclideanDistanceMatrix[0], &bestRotationMatrix[0],
            inputData_.som_size, &som_[0], inputData_.numberOfChannels * inputData_.neuron_size,
            inputData_.numberOfRotationsAndFlip, &rotatedImages[0]);

        resultFile.write((char*)&euclideanDistanceMatrix[0], inputData_.som_size * sizeof(float));

        if (inputData_.write_rot_flip) {
        	for (int i = 0; i != inputData_.som_size; ++i) {
        		char flip = bestRotationMatrix[i] / inputData_.numberOfRotations;
        		float angle = (bestRotationMatrix[i] % inputData_.numberOfRotations) * angleStepRadians;
        	    write_rot_flip_file.write(&flip, sizeof(char));
        	    write_rot_flip_file.write((char*)&angle, sizeof(float));
        	}
        }
    }

    std::cout << "  Progress: " << std::setw(12) << updateCount << " updates, 100 % ("
         << std::chrono::duration_cast<std::chrono::seconds>(myclock::now() - startTime).count() << " s)" << std::endl;
}

} // namespace pink
