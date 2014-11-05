/**
 * @file   EuclideanDistanceTest.cpp
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/ImageProcessing.h"
#include "SelfOrganizingMapLib/SelfOrganizingMap.h"
#include "CudaLib/CudaLib.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <vector>

const float FLOAT_INEQUALITY_TOLERANCE = float(1.0 / (1 << 22));

template <class T>
::testing::AssertionResult EqualFloatArrays(
                                const T* const expected,
                                const T* const actual,
                                unsigned long length)
{
    ::testing::AssertionResult result = ::testing::AssertionFailure();
    int errorsFound = 0;
    const char* separator = " ";
    for (unsigned long index = 0; index < length; index++)
    {
        if (fabs(expected[index] - actual[index]) > FLOAT_INEQUALITY_TOLERANCE)
        {
            if (errorsFound == 0)
            {
                result << "Differences found:";
            }
            if (errorsFound < 3)
            {
                result << separator
                        << expected[index] << " != " << actual[index]
                        << " @ " << index;
                separator = ", ";
            }
            errorsFound++;
        }
    }
    if (errorsFound > 0)
    {
        result << separator << errorsFound << " differences in total";
        return result;
    }
    return ::testing::AssertionSuccess();
}

TEST(EuclideanDistanceTest, Array)
{
	int length = 72;
	float *a = new float[length];
	fillRandom(a, length, 0);
	float *b = new float[length];
	fillRandom(b, length, 1);

	float cpu_result = calculateEuclideanDistanceWithoutSquareRoot(a, b, length);
	float gpu_result = cuda_calculateEuclideanDistanceWithoutSquareRoot(a, b, length);

	EXPECT_FLOAT_EQ(cpu_result, gpu_result);

	delete [] b;
	delete [] a;
}

TEST(EuclideanDistanceTest, SOM)
{
	int som_dim = 3;
	int som_size = som_dim * som_dim;
	int image_dim = 64;
	int image_size = image_dim * image_dim;
	int num_rot = 3;

	float *som = new float[som_size * image_size];
	float *rotatedImages = new float[num_rot * image_size];
	float *cpu_euclideanDistanceMatrix = new float[som_size];
	int *cpu_bestRotationMatrix = new int[som_size];

	fillRandom(som, som_size * image_size, 0);
	fillRandom(rotatedImages, num_rot * image_size, 1);

	generateEuclideanDistanceMatrix(cpu_euclideanDistanceMatrix, cpu_bestRotationMatrix, som_dim, som,
	    image_dim, num_rot, rotatedImages);

	float *d_som = cuda_alloc_float(som_size * image_size);
	float *d_rotatedImages = cuda_alloc_float(num_rot * image_size);
	float *d_euclideanDistanceMatrix = cuda_alloc_float(som_size);
	int *d_bestRotationMatrix = cuda_alloc_int(som_size);

	cuda_copyHostToDevice_float(som, d_som, som_size * image_size);
	cuda_copyHostToDevice_float(rotatedImages, d_rotatedImages, num_rot * image_size);

	cuda_generateEuclideanDistanceMatrix_algo2(d_euclideanDistanceMatrix, d_bestRotationMatrix, som_dim, d_som,
	    image_dim, num_rot, d_rotatedImages);

	float *gpu_euclideanDistanceMatrix = new float[som_size];
	int *gpu_bestRotationMatrix = new int[som_size];

	cuda_copyDeviceToHost_float(d_euclideanDistanceMatrix, gpu_euclideanDistanceMatrix, som_size);
	cuda_copyDeviceToHost_int(d_bestRotationMatrix, gpu_bestRotationMatrix, som_size);

	EXPECT_TRUE(EqualFloatArrays(cpu_euclideanDistanceMatrix, gpu_euclideanDistanceMatrix, som_size));
	EXPECT_TRUE(EqualFloatArrays(cpu_bestRotationMatrix, gpu_bestRotationMatrix, som_size));

	delete [] rotatedImages;
	delete [] som;
}
