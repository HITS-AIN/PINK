/**
 * @file   UtilitiesTest/DistributionFunctorTest.cpp
 * @date   Nov 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>

#include "UtilitiesLib/DistributionFunctor.h"

using namespace pink;

struct DistributionFunctorTestData
{
    DistributionFunctorTestData(float sigma, float damping)
     : sigma(sigma),
       damping(damping)
    {}

    float sigma;
    float damping;
};

class FullDistributionFunctorTest : public ::testing::TestWithParam<DistributionFunctorTestData>
{};

TEST(DistributionFunctorTest, GaussianSpecial)
{
    GaussianFunctor gauss(1.2f, 1.0f);

    EXPECT_NEAR(gauss(9.0), 2.028607587901271e-13, 1e-6);
    EXPECT_NEAR(gauss(10.0), 2.7673267835957437e-16, 1e-6);
}

TEST_P(FullDistributionFunctorTest, Gaussian)
{
    GaussianFunctor gauss(GetParam().sigma, GetParam().damping);

    // max value
    EXPECT_NEAR(gauss(0.0f), 1.0 / (GetParam().sigma * std::sqrt(2.0 * M_PI)), 1e-6);

    // inflection points
    EXPECT_NEAR(gauss(GetParam().sigma), 1.0 / (GetParam().sigma * std::sqrt(2.0 * M_PI * std::exp(1))), 1e-6);
    EXPECT_NEAR(gauss(-GetParam().sigma), 1.0 / (GetParam().sigma * std::sqrt(2.0 * M_PI * std::exp(1))), 1e-6);
}

TEST_P(FullDistributionFunctorTest, MexicanHat)
{
    try {
        MexicanHatFunctor mexican_hat(GetParam().sigma, GetParam().damping);

        // max value
        EXPECT_NEAR(mexican_hat(0.0f), 2.0 / (std::sqrt(3.0 * GetParam().sigma * std::sqrt(M_PI))), 1e-6);

        // value at x = +/- 1
        float sigma2 = GetParam().sigma * GetParam().sigma;
        EXPECT_NEAR(mexican_hat(1.0f),
            2.0 / (std::sqrt(3.0 * GetParam().sigma * std::sqrt(M_PI)))
                * (1.0 - 1.0 / sigma2) * std::exp(-1.0 / (2.0 * sigma2)), 1e-6);
        EXPECT_NEAR(mexican_hat(-1.0),
            2.0 / (std::sqrt(3.0 * GetParam().sigma * std::sqrt(M_PI)))
                * (1.0 - 1.0 / sigma2) * std::exp(-1.0 / (2.0 * sigma2)), 1e-6);
    } catch ( ... ) {
        if (GetParam().sigma > 0.0) FAIL() << "Exception with sigma > 0.0";
    }
}

INSTANTIATE_TEST_CASE_P(FullDistributionFunctorTest_all, FullDistributionFunctorTest,
    ::testing::Values(
        DistributionFunctorTestData( 1.0f, 1.0f),
        DistributionFunctorTestData( 1.2f, 1.0f),
        DistributionFunctorTestData( 2.0f, 1.0f),
        DistributionFunctorTestData(-2.1f, 1.0f)
));
