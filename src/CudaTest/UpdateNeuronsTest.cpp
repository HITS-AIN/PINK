/**
 * @file   CudaTest/UpdateNeuronsTest.cpp
 * @date   Oct 14, 2014
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

struct FullUpdateNeuronsTestData
{
	FullUpdateNeuronsTestData(int som_dim, int neuron_dim, int num_rot)
	  : som_dim(som_dim), neuron_dim(neuron_dim), num_rot(num_rot)
	{
		som_size = som_dim * som_dim;
        neuron_size = neuron_dim * neuron_dim;
		rot_size = num_rot * neuron_size;
		som_total_size = som_size * neuron_size;
	}

	int som_dim;
	int neuron_dim;
	int num_rot;

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
	float *cpu_som = new float[data.som_total_size];
	fillWithRandomNumbers(cpu_som, data.som_total_size, 1);
	int *bestRotationMatrix = new int[data.som_size];
	fillWithValue(bestRotationMatrix, data.som_size);
	float *euclideanDistanceMatrix = new float[data.som_size];
	fillWithValue(euclideanDistanceMatrix, data.som_size);

	updateNeurons(data.som_dim, cpu_som, data.neuron_dim, rotatedImages, Point(0,0), bestRotationMatrix, 1);

	float *gpu_som = new float[data.som_total_size];
	fillWithRandomNumbers(gpu_som, data.som_total_size, 1);

	float *d_rotatedImages = cuda_alloc_float(data.rot_size);
	float *d_som = cuda_alloc_float(data.som_total_size);
	int *d_bestRotationMatrix = cuda_alloc_int(data.som_size);
	float *d_euclideanDistanceMatrix = cuda_alloc_float(data.som_size);
    int *d_bestMatch = cuda_alloc_int(2);

	cuda_copyHostToDevice_float(d_rotatedImages, rotatedImages, data.rot_size);
	cuda_copyHostToDevice_float(d_som, gpu_som, data.som_total_size);
	cuda_copyHostToDevice_int(d_bestRotationMatrix, bestRotationMatrix, data.som_size);
	cuda_copyHostToDevice_float(d_euclideanDistanceMatrix, euclideanDistanceMatrix, data.som_size);

	cuda_updateNeurons(d_som, d_rotatedImages, d_bestRotationMatrix, d_euclideanDistanceMatrix, d_bestMatch,
	    data.som_dim, data.neuron_dim, data.num_rot);

	cuda_copyDeviceToHost_float(gpu_som, d_som, data.som_total_size);

	EXPECT_TRUE(EqualFloatArrays(cpu_som, gpu_som, data.som_total_size));

	cuda_free(d_euclideanDistanceMatrix);
	cuda_free(d_bestRotationMatrix);
	cuda_free(d_som);
	cuda_free(d_rotatedImages);
	cuda_free(d_bestMatch);

	delete [] euclideanDistanceMatrix;
	delete [] bestRotationMatrix;
	delete [] gpu_som;
	delete [] cpu_som;
	delete [] rotatedImages;
}

INSTANTIATE_TEST_CASE_P(FullUpdateNeuronsTest_all, FullUpdateNeuronsTest,
    ::testing::Values(
        FullUpdateNeuronsTestData(2,2,2)
));
