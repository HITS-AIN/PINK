/**
 * @file   ImageProcessingTest/ImageProcessingTest.cpp
 * @brief  Unit tests for image processing.
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "UtilitiesLib/Filler.h"
#include "gtest/gtest.h"
#include <cmath>
#include <vector>

using namespace std;
using namespace PINK;

TEST(ImageProcessingTest, Rotation90)
{
	int height = 4;
	int width = 3;
	int size = height * width;
	vector<float> image(size), image2(size), image3(size);
	fillWithRandomNumbers(&image[0], size);

	// 4 times rotating by 90 degrees should result in original image
	rotate_90degrees(height, width, &image[0], &image2[0]);
	rotate_90degrees(width, height, &image2[0], &image3[0]);
	rotate_90degrees(height, width, &image3[0], &image2[0]);
	rotate_90degrees(width, height, &image2[0], &image3[0]);

	EXPECT_TRUE(EqualFloatArrays(&image[0], &image3[0], size));
}

TEST(ImageProcessingTest, CompareRotation90WithRotation)
{
	int height = 13;
	int width = 13;
	int size = height * width;
	vector<float> image(size), image2(size), image3(size);
	fillWithRandomNumbers(&image[0], size);

	rotate_90degrees(height, width, &image[0], &image2[0]);
	rotate(height, width, &image[0], &image3[0], 0.5*M_PI, NEAREST_NEIGHBOR);

	EXPECT_TRUE(EqualFloatArrays(&image2[0], &image3[0], size));
}

TEST(ImageProcessingTest, BilinearInterpolation)
{
	int height = 12;
	int width = 12;
	int size = height * width;
	vector<float> image(size), image2(size), image3(size);
	fillWithRandomNumbers(&image[0], size);

	rotate_90degrees(height, width, &image[0], &image2[0]);
	//printImage(&image2[0], width, height);
	rotate(height, width, &image[0], &image3[0], 0.5*M_PI, BILINEAR);
	//printImage(&image3[0], width, height);

	EXPECT_TRUE(EqualFloatArrays(&image2[0], &image3[0], size));
}

TEST(ImageProcessingTest, EuclideanSimilarity)
{
	std::vector<float> a{2.0, -3.9, 0.1};
	std::vector<float> b{1.9, -4.0, 0.2};

	EXPECT_NEAR(0.1732 ,(calculateEuclideanDistance(&a[0], &b[0], a.size())), 1e-4);
}

TEST(ImageProcessingTest, Flip)
{
	int dim = 3;
	int size = dim*dim;

	std::vector<float> va(size);
	float *a = &va[0];
	fillWithRandomNumbers(a,size);

	std::vector<float> vb(size);
	float *b = &vb[0];
	flip(dim,dim,a,b);

	std::vector<float> vc(size);
	float *c = &vc[0];
	flip(dim,dim,b,c);

	EXPECT_EQ(va, vc);
}

TEST(ImageProcessingTest, Crop)
{
	int dim = 4;
	int size = dim * dim;
	int crop_dim = 2;
	int crop_size = crop_dim * crop_dim;

	std::vector<float> va(size);
	float *a = &va[0];
	fillWithRandomNumbers(a,size);

	std::vector<float> vb(crop_size);
	float *b = &vb[0];
	crop(dim,dim,crop_dim,crop_dim,a,b);

	float suma = a[5] + a[6] + a[9] + a[10];
	float sumb = b[0] + b[1] + b[2] + b[3];

	EXPECT_FLOAT_EQ(suma, sumb);
}

TEST(ImageProcessingTest, FlipAndCrop)
{
	int dim = 4;
	int size = dim * dim;
	int crop_dim = 2;
	int crop_size = crop_dim * crop_dim;

	std::vector<float> va(size);
	float *a = &va[0];
	fillWithRandomNumbers(a,size);

	std::vector<float> vb(crop_size);
	float *b = &vb[0];
	flipAndCrop(dim,dim,crop_dim,crop_dim,a,b);

	std::vector<float> vc(crop_size);
	float *c = &vc[0];
	flip(crop_dim,crop_dim,b,c);

	float suma = a[5] + a[6] + a[9] + a[10];
	float sumb = b[0] + b[1] + b[2] + b[3];

	EXPECT_FLOAT_EQ(suma, sumb);
}

TEST(ImageProcessingTest, RotateAndCrop)
{
	int dim = 4;
	int size = dim * dim;
	int crop_dim = 2;
	int crop_size = crop_dim * crop_dim;

	std::vector<float> va(size);
	float *a = &va[0];
	fillWithRandomNumbers(a,size);

	std::vector<float> vb(crop_size);
	float *b = &vb[0];
	rotateAndCrop(dim, dim, crop_dim, crop_dim, a, b, 2.0*M_PI);

	float suma = a[5] + a[6] + a[9] + a[10];
	float sumb = b[0] + b[1] + b[2] + b[3];

	EXPECT_FLOAT_EQ(suma, sumb);
}
