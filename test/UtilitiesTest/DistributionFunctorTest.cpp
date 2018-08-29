/**
 * @file   UtilitiesTest/DistributionFunctorTest.cpp
 * @date   Nov 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/DistributionFunctor.h"
#include "gtest/gtest.h"
#include <cmath>
#include <string>

using namespace pink;

struct DistributionFunctorTestData
{
    DistributionFunctorTestData(float sigma) : sigma(sigma) {}

    float sigma;
};

class FullDistributionFunctorTest : public ::testing::TestWithParam<DistributionFunctorTestData>
{};

TEST(DistributionFunctorTest, GaussianSpecial)
{
    EXPECT_NEAR(GaussianFunctor(1.2)(9.0), 2.028607587901271e-13, 1e-6);
    EXPECT_NEAR(GaussianFunctor(1.2)(10.0), 2.7673267835957437e-16, 1e-6);
}

TEST_P(FullDistributionFunctorTest, Gaussian)
{
    // max value
    EXPECT_NEAR(GaussianFunctor(GetParam().sigma)(0.0), 1.0 / (GetParam().sigma * sqrt(2.0*M_PI)), 1e-6);

    // inflection points
    EXPECT_NEAR(GaussianFunctor(GetParam().sigma)(GetParam().sigma), 1.0 / (GetParam().sigma * sqrt(2.0*M_PI*exp(1))), 1e-6);
    EXPECT_NEAR(GaussianFunctor(GetParam().sigma)(-GetParam().sigma), 1.0 / (GetParam().sigma * sqrt(2.0*M_PI*exp(1))), 1e-6);
}

TEST_P(FullDistributionFunctorTest, MexicanHat)
{
//    MexicanHatFunctor f(GetParam().sigma);
//    for (float x = -4.0; x < 4.0; x+= 0.05) std::cout << x << "    " << f(x) << std::endl;

    try {
        // max value
        EXPECT_NEAR(MexicanHatFunctor(GetParam().sigma)(0.0), 2.0 / (sqrt(3.0 * GetParam().sigma * sqrt(M_PI))), 1e-6);

        // value at x = +/- 1
        float sigma2 = GetParam().sigma * GetParam().sigma;
        EXPECT_NEAR(MexicanHatFunctor(GetParam().sigma)(1.0),
            2.0 / (sqrt(3.0 * GetParam().sigma * sqrt(M_PI))) * (1.0 - 1.0 / sigma2) * exp(-1.0 / (2.0 * sigma2)), 1e-6);
        EXPECT_NEAR(MexicanHatFunctor(GetParam().sigma)(-1.0),
            2.0 / (sqrt(3.0 * GetParam().sigma * sqrt(M_PI))) * (1.0 - 1.0 / sigma2) * exp(-1.0 / (2.0 * sigma2)), 1e-6);
    } catch ( ... ) {
        if (GetParam().sigma > 0.0) FAIL() << "Exception with sigma > 0.0";
    }
}

INSTANTIATE_TEST_CASE_P(FullDistributionFunctorTest_all, FullDistributionFunctorTest,
    ::testing::Values(
        DistributionFunctorTestData(1.0),
        DistributionFunctorTestData(1.2),
        DistributionFunctorTestData(2.0),
        DistributionFunctorTestData(-2.1)
));
