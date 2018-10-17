/**
 * @file   CudaTest/UpdateNeuronsTest.cpp
 * @date   Oct 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib/CudaLib.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/Filler.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace pink;

struct FullUpdateNeuronsTestData
{
    FullUpdateNeuronsTestData(int som_dim, int neuron_dim, int num_rot, int num_channels)
      : som_dim(som_dim), neuron_dim(neuron_dim), num_rot(num_rot), num_channels(num_channels)
    {
        som_size = som_dim * som_dim;
        neuron_size = neuron_dim * neuron_dim;
        rot_size = num_channels * num_rot * neuron_size;
        som_total_size = num_channels * som_size * neuron_size;
    }

    int som_dim;
    int neuron_dim;
    int num_rot;
    int num_channels;

    int som_size;
    int neuron_size;
    int rot_size;
    int som_total_size;
};

class FullUpdateNeuronsTest : public ::testing::TestWithParam<FullUpdateNeuronsTestData>
{};

//! Compare rotated images between CPU and GPU version.
TEST_P(FullUpdateNeuronsTest, UpdateNeurons)
{
    FullUpdateNeuronsTestData data = GetParam();

    float *rotatedImages = new float[data.rot_size];
    fillWithRandomNumbers(rotatedImages, data.rot_size, 0);
    int *bestRotationMatrix = new int[data.som_size];
    fillWithValue(bestRotationMatrix, data.som_size);
    float *euclideanDistanceMatrix = new float[data.som_size];
    fillWithValue(euclideanDistanceMatrix, data.som_size);

    InputData inputData;
    inputData.som_width = data.som_dim;
    inputData.som_height = data.som_dim;
    inputData.dimensionality = 2;
    inputData.neuron_dim = data.neuron_dim;
    inputData.numberOfRotations = data.num_rot;
    inputData.numberOfImages = 1;
    inputData.numberOfChannels = data.num_channels;
    inputData.image_dim = data.neuron_dim;
    inputData.image_size = data.neuron_size;
    inputData.som_size = data.som_size;
    inputData.neuron_size = data.neuron_size;
    inputData.som_total_size = data.som_total_size;
    inputData.numberOfRotationsAndFlip = data.num_rot;

    SOM cpu_som(inputData);
    std::vector<float> gpu_som = cpu_som.getData();

    // Calculate CPU result
    cpu_som.updateNeurons(rotatedImages, 0, bestRotationMatrix);

    float *d_rotatedImages = cuda_alloc_float(data.rot_size);
    float *d_som = cuda_alloc_float(data.som_total_size);
    int *d_bestRotationMatrix = cuda_alloc_int(data.som_size);
    float *d_euclideanDistanceMatrix = cuda_alloc_float(data.som_size);
    int *d_bestMatch = cuda_alloc_int(1);

    cuda_copyHostToDevice_float(d_rotatedImages, rotatedImages, data.rot_size);
    cuda_copyHostToDevice_float(d_som, &gpu_som[0], data.som_total_size);
    cuda_copyHostToDevice_int(d_bestRotationMatrix, bestRotationMatrix, data.som_size);
    cuda_copyHostToDevice_float(d_euclideanDistanceMatrix, euclideanDistanceMatrix, data.som_size);

    // Calculate GPU result
    update_neurons(d_som, d_rotatedImages, d_bestRotationMatrix, d_euclideanDistanceMatrix, d_bestMatch,
        data.som_dim, data.som_dim, 1, data.som_size, data.num_channels * data.neuron_size,
        DistributionFunction::GAUSSIAN, Layout::CARTESIAN, DEFAULT_SIGMA, DEFAULT_DAMPING, inputData.max_update_distance,
        inputData.usePBC, inputData.dimensionality);

    cuda_copyDeviceToHost_float(&gpu_som[0], d_som, data.som_total_size);

    int bestMatch;
    cuda_copyDeviceToHost_int(&bestMatch, d_bestMatch, 1);
    EXPECT_EQ(bestMatch,0);

    EXPECT_TRUE(EqualFloatArrays(cpu_som.getDataPointer(), &gpu_som[0], data.som_total_size));

    cuda_free(d_euclideanDistanceMatrix);
    cuda_free(d_bestRotationMatrix);
    cuda_free(d_som);
    cuda_free(d_rotatedImages);
    cuda_free(d_bestMatch);

    delete [] euclideanDistanceMatrix;
    delete [] bestRotationMatrix;
    delete [] rotatedImages;
}

INSTANTIATE_TEST_CASE_P(FullUpdateNeuronsTest_all, FullUpdateNeuronsTest,
    ::testing::Values(
        FullUpdateNeuronsTestData(2,2,1,1),
        FullUpdateNeuronsTestData(2,2,1,2),
        FullUpdateNeuronsTestData(2,2,2,1),
        FullUpdateNeuronsTestData(2,2,2,2)
));
