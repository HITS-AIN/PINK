/**
 * @file   ImageProcessingTest/ImageProcessingTest.cpp
 * @brief  Unit tests for image processing.
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "ImageProcessingLib/Image.h"
#include "ImageProcessingLib/ImageProcessing.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "UtilitiesLib/Filler.h"

using namespace pink;

TEST(ImageProcessingTest, Rotation90)
{
    uint32_t height = 4;
    uint32_t width = 3;
    uint32_t size = height * width;
    std::vector<float> image(size), image2(size), image3(size);
    fill_random_uniform(&image[0], size);

    // 4 times rotating by 90 degrees should result in original image
    rotate_90degrees(height, width, &image[0], &image2[0]);
    rotate_90degrees(width, height, &image2[0], &image3[0]);
    rotate_90degrees(height, width, &image3[0], &image2[0]);
    rotate_90degrees(width, height, &image2[0], &image3[0]);

    EXPECT_TRUE(EqualFloatArrays(&image[0], &image3[0], size));
}

TEST(ImageProcessingTest, CompareRotation90WithRotation)
{
    uint32_t height = 13;
    uint32_t width = 13;
    uint32_t size = height * width;
    std::vector<float> image(size), image2(size), image3(size);
    fill_random_uniform(&image[0], size);

    rotate_90degrees(height, width, &image[0], &image2[0]);
    rotate(height, width, &image[0], &image3[0], 0.5*M_PI, Interpolation::NEAREST_NEIGHBOR);

    EXPECT_TRUE(EqualFloatArrays(&image2[0], &image3[0], size));
}

TEST(ImageProcessingTest, BilinearInterpolation)
{
    uint32_t height = 3;
    uint32_t width = 3;
    uint32_t size = height * width;
    std::vector<float> image(size), image2(size), image3(size);
    fill_random_uniform(&image[0], size);

    rotate_90degrees(height, width, &image[0], &image2[0]);
    rotate(height, width, &image[0], &image3[0], 0.5*M_PI, Interpolation::BILINEAR);

    EXPECT_TRUE(EqualFloatArrays(&image2[0], &image3[0], size, 1e-4));
}

TEST(ImageProcessingTest, EuclideanSimilarity)
{
    const std::vector<float> a{2.0, -3.9, 0.1};
    const std::vector<float> b{1.9, -4.0, 0.2};

    EXPECT_NEAR(0.1732, (euclidean_distance(&a[0], &b[0], a.size())), 1e-4);
}

TEST(ImageProcessingTest, EuclideanDistanceByDot)
{
    std::vector<float> a{2.0, -3.9, 0.1};
    std::vector<float> b{1.9, -4.0, 0.2};
    std::vector<float> c;

    std::transform(begin(a), end(a), begin(b), back_inserter(c), std::minus<float>());

    auto&& dot = std::accumulate(begin(c), end(c), 0.0, [] (float dot, float c) {
        return dot + std::pow(c, 2);
    });

    EXPECT_NEAR(0.1732, std::sqrt(dot), 1e-4);
}

// Flip direction is left-right
TEST(ImageProcessingTest, flip)
{
    std::vector<float> a{1, 2, 3, 4};

    std::vector<float> b(4);
    flip(2, 2, &a[0], &b[0]);

    std::vector<float> c{3, 4, 1, 2};
    EXPECT_EQ(c, b);
}

// Check double flip invariance
TEST(ImageProcessingTest, double_flip)
{
    std::vector<float> a{1, 2, 3, 4};

    std::vector<float> b(4);
    flip(2, 2, &a[0], &b[0]);

    std::vector<float> c(4);
    flip(2, 2, &b[0], &c[0]);

    EXPECT_EQ(a, c);
}

TEST(ImageProcessingTest, Crop)
{
    uint32_t dim = 4;
    uint32_t size = dim * dim;
    uint32_t crop_dim = 2;
    uint32_t crop_size = crop_dim * crop_dim;

    std::vector<float> va(size);
    float *a = &va[0];
    fill_random_uniform(a,size);

    std::vector<float> vb(crop_size);
    float *b = &vb[0];
    crop(dim,dim,crop_dim,crop_dim,a,b);

    float suma = a[5] + a[6] + a[9] + a[10];
    float sumb = b[0] + b[1] + b[2] + b[3];

    EXPECT_FLOAT_EQ(suma, sumb);
}

TEST(ImageProcessingTest, FlipAndCrop)
{
    uint32_t dim = 4;
    uint32_t size = dim * dim;
    uint32_t crop_dim = 2;
    uint32_t crop_size = crop_dim * crop_dim;

    std::vector<float> va(size);
    float *a = &va[0];
    fill_random_uniform(a,size);

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
    uint32_t dim = 4;
    uint32_t size = dim * dim;
    uint32_t crop_dim = 2;
    uint32_t crop_size = crop_dim * crop_dim;

    std::vector<float> va(size);
    float *a = &va[0];
    fill_random_uniform(a,size);

    std::vector<float> vb(crop_size);
    float *b = &vb[0];
    rotateAndCrop(dim, dim, crop_dim, crop_dim, a, b, 2.0*M_PI);

    float suma = a[5] + a[6] + a[9] + a[10];
    float sumb = b[0] + b[1] + b[2] + b[3];

    EXPECT_FLOAT_EQ(suma, sumb);
}
