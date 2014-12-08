/**
 * @file   CudaTest/EuclideanDistanceTest.cpp
 * @brief  Unit tests for calculating euclidean distance using CUDA.
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib/CudaLib.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "UtilitiesLib/Filler.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <vector>

//! Compare squared euclidean distance between CPU and GPU version.
TEST(EuclideanDistanceTest, Array)
{
	int length = 72;
	float *a = new float[length];
	fillWithRandomNumbers(a, length, 0);
	float *b = new float[length];
	fillWithRandomNumbers(b, length, 1);

	float cpu_result = calculateEuclideanDistanceWithoutSquareRoot(a, b, length);
	float gpu_result = cuda_calculateEuclideanDistanceWithoutSquareRoot(a, b, length);

	EXPECT_FLOAT_EQ(cpu_result, gpu_result);

	delete [] b;
	delete [] a;
}

struct FullEuclideanDistanceTestData
{
    FullEuclideanDistanceTestData(int som_dim, int neuron_dim, int num_rot, int num_channels)
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

class FullEuclideanDistanceTest : public ::testing::TestWithParam<FullEuclideanDistanceTestData>
{};

//! Compare squared euclidean distance matrix between CPU and GPU version.
TEST_P(FullEuclideanDistanceTest, cuda_generateEuclideanDistanceMatrix)
{
    FullEuclideanDistanceTestData data = GetParam();

	float *som = new float[data.som_total_size];
	float *rotatedImages = new float[data.rot_size];
	float *cpu_euclideanDistanceMatrix = new float[data.som_size];
	int *cpu_bestRotationMatrix = new int[data.som_size];

	fillWithRandomNumbers(som, data.som_total_size, 0);
	fillWithRandomNumbers(rotatedImages, data.rot_size, 1);

	generateEuclideanDistanceMatrix(cpu_euclideanDistanceMatrix, cpu_bestRotationMatrix, data.som_dim, som,
	    data.neuron_dim, data.num_rot, rotatedImages, data.num_channels);

	float *d_som = cuda_alloc_float(data.som_total_size);
	float *d_rotatedImages = cuda_alloc_float(data.rot_size);
	float *d_euclideanDistanceMatrix = cuda_alloc_float(data.som_size);
	int *d_bestRotationMatrix = cuda_alloc_int(data.som_size);

	cuda_copyHostToDevice_float(d_som, som, data.som_total_size);
	cuda_copyHostToDevice_float(d_rotatedImages, rotatedImages, data.rot_size);

	cuda_generateEuclideanDistanceMatrix(d_euclideanDistanceMatrix, d_bestRotationMatrix, data.som_dim, d_som,
	    data.neuron_dim, data.num_rot, d_rotatedImages, data.num_channels);

	float *gpu_euclideanDistanceMatrix = new float[data.som_size];
	int *gpu_bestRotationMatrix = new int[data.som_size];

	cuda_copyDeviceToHost_float(gpu_euclideanDistanceMatrix, d_euclideanDistanceMatrix, data.som_size);
	cuda_copyDeviceToHost_int(gpu_bestRotationMatrix, d_bestRotationMatrix, data.som_size);

//	for (int i = 0; i < data.som_size; ++i)
//	    std::cout << cpu_euclideanDistanceMatrix[i] << " " << gpu_euclideanDistanceMatrix[i] << std::endl;

	EXPECT_TRUE(EqualFloatArrays(cpu_euclideanDistanceMatrix, gpu_euclideanDistanceMatrix, data.som_size, 1.0e-3));
	EXPECT_TRUE(EqualFloatArrays(cpu_bestRotationMatrix, gpu_bestRotationMatrix, data.som_size, 1.0e-3));

	delete [] gpu_bestRotationMatrix;
	delete [] gpu_euclideanDistanceMatrix;
	delete [] cpu_bestRotationMatrix;
	delete [] cpu_euclideanDistanceMatrix;
	delete [] rotatedImages;
	delete [] som;
}

INSTANTIATE_TEST_CASE_P(FullEuclideanDistanceTest_all, FullEuclideanDistanceTest,
    ::testing::Values(
        FullEuclideanDistanceTestData( 2,  3,   2, 1),
        FullEuclideanDistanceTestData(10, 44, 180, 1),
        FullEuclideanDistanceTestData(10, 44, 180, 2)
));
