/**
 * @file   SelfOrganizingMapTest/TrainerCompare.cpp
 * @brief  Unit tests for image processing.
 * @date   Nov 5, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <omp.h>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/DataIO.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "SelfOrganizingMapLib/SOMIO.h"
#include "SelfOrganizingMapLib/Trainer.h"
#include "SelfOrganizingMapLib/Trainer_generic.h"
#include "UtilitiesLib/EqualFloatArrays.h"
#include "UtilitiesLib/DistributionFunctor.h"


using namespace pink;

struct TrainerCompareTestData
{
    TrainerCompareTestData(int num_rot)
     : num_rot(num_rot)
    {}

    int num_rot;
};

class TrainerCompareTest : public ::testing::TestWithParam<TrainerCompareTestData>
{};

TEST_P(TrainerCompareTest, cartesian_2d_float)
{
    typedef Data<CartesianLayout<2>, float> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false> MyTrainer;
    typedef Trainer_generic<CartesianLayout<2>, CartesianLayout<2>, float, false> MyTrainer_generic;

    uint32_t som_dim = 2;
    uint32_t image_dim = 2;
    uint32_t neuron_dim = 2;

    DataType data({image_dim, image_dim}, {1, 2, 3, 4});
    SOMType som1({som_dim, som_dim}, {neuron_dim, neuron_dim}, std::vector<float>(16, 0.0));
    SOMType som2({som_dim, som_dim}, {neuron_dim, neuron_dim}, std::vector<float>(16, 0.0));

    auto&& f = GaussianFunctor(1.1, 0.2);

    MyTrainer trainer1(som1, f, 0, GetParam().num_rot, false, 0.0, Interpolation::BILINEAR);
    trainer1(data);

    MyTrainer_generic trainer2(som2, f, 0, GetParam().num_rot, false, 0.0, Interpolation::BILINEAR);
    trainer2(data);

    std::cout << som1.get_data().size() << std::endl;

    EXPECT_EQ(som1.get_data().size(), som2.get_data().size());
    EXPECT_TRUE(EqualFloatArrays(som1.get_data_pointer(), som2.get_data_pointer(), som1.get_data().size(), 1e-4));
}

INSTANTIATE_TEST_CASE_P(TrainerCompareTest_all, TrainerCompareTest,
    ::testing::Values(
        TrainerCompareTestData(1),
        TrainerCompareTestData(4)
));
