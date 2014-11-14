/**
 * @file   CudaTest/EuclideanDistanceTest.cpp
 * @brief  Unit tests for calculating euclidean distance using CUDA.
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib/CudaLib.h"
#include "EqualFloatArrays.h"
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

//! Compare squared euclidean distance matrix (algo 2) between CPU and GPU version.
TEST(EuclideanDistanceTest, cuda_generateEuclideanDistanceMatrix_algo2)
{
	int som_dim = 2;
	int image_dim = 3;
	int num_rot = 2;
	int som_size = som_dim * som_dim;
	int image_size = image_dim * image_dim;

	float *som = new float[som_size * image_size];
	float *rotatedImages = new float[num_rot * image_size];
	float *cpu_euclideanDistanceMatrix = new float[som_size];
	int *cpu_bestRotationMatrix = new int[som_size];

	fillWithRandomNumbers(som, som_size * image_size, 0);
	fillWithRandomNumbers(rotatedImages, num_rot * image_size, 1);

	generateEuclideanDistanceMatrix(cpu_euclideanDistanceMatrix, cpu_bestRotationMatrix, som_dim, som,
	    image_dim, num_rot, rotatedImages);

//	for (int i=0; i < som_size; ++i) {
//		std::cout << "cpu eucl " << i << ": " << calculateEuclideanDistanceWithoutSquareRoot(som + i*image_size, rotatedImages, image_size) << std::endl;
//    }

	float *d_som = cuda_alloc_float(som_size * image_size);
	float *d_rotatedImages = cuda_alloc_float(num_rot * image_size);
	float *d_euclideanDistanceMatrix = cuda_alloc_float(som_size);
	int *d_bestRotationMatrix = cuda_alloc_int(som_size);

	cuda_copyHostToDevice_float(d_som, som, som_size * image_size);
	cuda_copyHostToDevice_float(d_rotatedImages, rotatedImages, num_rot * image_size);

	cuda_generateEuclideanDistanceMatrix_algo2(d_euclideanDistanceMatrix, d_bestRotationMatrix, som_dim, d_som,
	    image_dim, num_rot, d_rotatedImages);

	float *gpu_euclideanDistanceMatrix = new float[som_size];
	int *gpu_bestRotationMatrix = new int[som_size];

	cuda_copyDeviceToHost_float(gpu_euclideanDistanceMatrix, d_euclideanDistanceMatrix, som_size);
	cuda_copyDeviceToHost_int(gpu_bestRotationMatrix, d_bestRotationMatrix, som_size);

	EXPECT_TRUE(EqualFloatArrays(cpu_euclideanDistanceMatrix, gpu_euclideanDistanceMatrix, som_size));
	EXPECT_TRUE(EqualFloatArrays(cpu_bestRotationMatrix, gpu_bestRotationMatrix, som_size));

	delete [] gpu_bestRotationMatrix;
	delete [] gpu_euclideanDistanceMatrix;
	delete [] cpu_bestRotationMatrix;
	delete [] cpu_euclideanDistanceMatrix;
	delete [] rotatedImages;
	delete [] som;
}
