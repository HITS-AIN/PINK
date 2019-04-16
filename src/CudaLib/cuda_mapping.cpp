/**
 * @file   CudaLib/cuda_mapping.cpp
 * @date   Dec 2, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageIterator.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/Error.h"
#include "UtilitiesLib/Filler.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace std;
using namespace chrono;

namespace pink {

void cuda_mapping(InputData const& inputData)
{
    cout << "  Starting CUDA version of mapping.\n" << endl;
    if (inputData.verbose) cuda_print_properties();

    // Open result file
    std::ofstream resultFile(inputData.resultFilename);
    if (!resultFile) fatalError("Error opening " + inputData.resultFilename);
    resultFile.write((char*)&inputData.numberOfImages, sizeof(int));
    resultFile.write((char*)&inputData.som_width, sizeof(int));
    resultFile.write((char*)&inputData.som_height, sizeof(int));
    resultFile.write((char*)&inputData.som_depth, sizeof(int));

    std::ofstream write_rot_flip_file;
    vector<int> bestRotationMatrix;
    if (inputData.write_rot_flip) {
        bestRotationMatrix.resize(inputData.som_size);
        write_rot_flip_file.open(inputData.rot_flip_filename);
        if (!write_rot_flip_file) fatalError("Error opening " + inputData.rot_flip_filename);
        write_rot_flip_file.write((char*)&inputData.numberOfImages, sizeof(int));
        write_rot_flip_file.write((char*)&inputData.som_width, sizeof(int));
        write_rot_flip_file.write((char*)&inputData.som_height, sizeof(int));
        write_rot_flip_file.write((char*)&inputData.som_depth, sizeof(int));
    }

    // Initialize SOM on host
    SOM som(inputData);

    // Initialize SOM on device
    float *d_som = cuda_alloc_float(som.getSize());
    cuda_copyHostToDevice_float(d_som, som.getDataPointer(), som.getSize());

    // Memory allocation
    int rotatedImagesSize = inputData.numberOfChannels * inputData.numberOfRotationsAndFlip * inputData.neuron_size;
    if (inputData.verbose) cout << "  Size of rotated images = " << rotatedImagesSize * sizeof(float)<< " bytes" << endl;
    float *d_rotatedImages = cuda_alloc_float(rotatedImagesSize);

    if (inputData.verbose) cout << "  Size of euclidean distance matrix = " << inputData.som_size * sizeof(float) << " bytes" << endl;
    float *d_euclideanDistanceMatrix = cuda_alloc_float(inputData.som_size);
    vector<float> euclideanDistanceMatrix(inputData.som_size);

    if (inputData.verbose) cout << "  Size of best rotation matrix = " << inputData.som_size * sizeof(int) << " bytes\n" << endl;
    int *d_bestRotationMatrix = cuda_alloc_int(inputData.som_size);

    if (inputData.verbose) cout << "  Size of image = " << inputData.numberOfChannels * inputData.image_size * sizeof(float) << " bytes\n" << endl;
    float *d_image = cuda_alloc_float(inputData.numberOfChannels * inputData.image_size);

    // Prepare trigonometric values
    float *d_cosAlpha = NULL, *d_sinAlpha = NULL;
    trigonometricValues(&d_cosAlpha, &d_sinAlpha, inputData.numberOfRotations/4);
    float angleStepRadians = 2.0 * M_PI / inputData.numberOfRotations;

    // Progress status
    float progress = 0.0;
    float progressStep = 1.0 / inputData.numberOfImages;
    float nextProgressPrint = inputData.progressFactor;
    int progressPrecision = rint(log10(1.0 / inputData.progressFactor)) - 2;
    if (progressPrecision < 0) progressPrecision = 0;

    // Start timer
    auto startTime = steady_clock::now();
    int updateCount = 0;

    for (ImageIterator<float> iterImage(inputData.imagesFilename),iterEnd; iterImage != iterEnd; ++iterImage, ++updateCount)
    {
        if ((inputData.progressFactor < 1.0 and progress > nextProgressPrint) or
            (inputData.progressFactor >= 1.0 and updateCount != 0 and !(updateCount % static_cast<int>(inputData.progressFactor))))
        {
            cout << "  Progress: " << setw(12) << updateCount << " updates, "
                 << fixed << setprecision(progressPrecision) << setw(3) << progress*100 << " % ("
                 << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;

            nextProgressPrint += inputData.progressFactor;
            startTime = steady_clock::now();
        }
        progress += progressStep;

        cuda_copyHostToDevice_float(d_image, iterImage->getPointerOfFirstPixel(), iterImage->getSize());

        generate_rotated_images_gpu(d_rotatedImages, d_image, inputData.numberOfRotations,
            inputData.image_dim, inputData.neuron_dim, inputData.useFlip, inputData.interpolation,
            d_cosAlpha, d_sinAlpha, inputData.numberOfChannels);

        cuda_generateEuclideanDistanceMatrix(d_euclideanDistanceMatrix, d_bestRotationMatrix,
            inputData.som_size, d_som, inputData.numberOfChannels * inputData.neuron_size,
            inputData.numberOfRotationsAndFlip, d_rotatedImages, inputData.block_size_1, inputData.useMultipleGPUs);

        cuda_copyDeviceToHost_float(&euclideanDistanceMatrix[0], d_euclideanDistanceMatrix, inputData.som_size);
        resultFile.write((char*)&euclideanDistanceMatrix[0], inputData.som_size * sizeof(float));

        if (inputData.write_rot_flip) {
            cuda_copyDeviceToHost_int(&bestRotationMatrix[0], d_bestRotationMatrix, inputData.som_size);
            for (int i = 0; i != inputData.som_size; ++i) {
                char flip = bestRotationMatrix[i] / inputData.numberOfRotations;
                float angle = (bestRotationMatrix[i] % inputData.numberOfRotations) * angleStepRadians;
                write_rot_flip_file.write(&flip, sizeof(char));
                write_rot_flip_file.write((char*)&angle, sizeof(float));
            }
        }
    }

    cout << "  Progress: " << setw(12) << updateCount << " updates, 100 % ("
         << duration_cast<seconds>(steady_clock::now() - startTime).count() << " s)" << endl;

    // Free memory
    if (d_cosAlpha) cuda_free(d_cosAlpha);
    if (d_sinAlpha) cuda_free(d_sinAlpha);
    cuda_free(d_image);
    cuda_free(d_bestRotationMatrix);
    cuda_free(d_euclideanDistanceMatrix);
    cuda_free(d_rotatedImages);
    cuda_free(d_som);
}

} // namespace pink
