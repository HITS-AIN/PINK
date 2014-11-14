/*
 * ImageProcessingTest.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "UtilitiesLib/Filler.h"
#include "gtest/gtest.h"
#include <cmath>


using namespace PINK;

TEST(ImageProcessingTest, Rotation)
{
	Image<float> image(10,10,1.0);
	Image<float> image2(image.getHeight(), image.getWidth());
	rotate(image.getHeight(), image.getWidth(), image.getPointerOfFirstPixel(), image2.getPointerOfFirstPixel(), 45.0*M_PI/180.0, NONE);

	std::vector<float> data{
		0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
		1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
		0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    };

	EXPECT_EQ(image2.getPixel(), data);
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
	float sumb = 0.0;
	for (int i = 0; i < crop_size; ++i) sumb += b[i];

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
	float sumb = 0.0;
	for (int i = 0; i < crop_size; ++i) sumb += b[i];

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

	float suma = a[1] + a[5] + a[6] + a[10];
	float sumb = 0.0;
	for (int i = 0; i < crop_size; ++i) sumb += b[i];

	EXPECT_FLOAT_EQ(suma, sumb);
}
