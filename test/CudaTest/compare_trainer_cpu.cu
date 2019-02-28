/**
 * @file   CudaTest/compare_trainer_cpu.cpp
 * @brief  Compare generic GPU trainer against generic CPU trainer.
 * @date   Nov 12, 2018
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
#include "UtilitiesLib/EqualFloatArrays.h"
#include "UtilitiesLib/DistributionFunctor.h"
#include "UtilitiesLib/Filler.h"

using namespace pink;

struct TrainerCompareTestData
{
    TrainerCompareTestData(uint32_t som_dim, uint32_t image_dim, uint32_t neuron_dim, int num_rot, bool use_flip)
     : som_dim(som_dim),
       image_dim(image_dim),
       neuron_dim(neuron_dim),
       num_rot(num_rot),
       use_flip(use_flip)
    {}

    uint32_t som_dim;
    uint32_t image_dim;
    uint32_t neuron_dim;

    uint32_t num_rot;
    bool use_flip;
};

class compare_trainer_cpu : public ::testing::TestWithParam<TrainerCompareTestData>
{};

TEST_P(compare_trainer_cpu, cartesian_2d_float)
{
    typedef Data<CartesianLayout<2>, float> DataContainerType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false> MyTrainer_cpu;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true> MyTrainer_gpu;

    DataContainerType data({GetParam().image_dim, GetParam().image_dim}, 0.0);
    fill_random_uniform(data.get_data_pointer(), data.size());

    SOMType som1({GetParam().som_dim, GetParam().som_dim}, {GetParam().neuron_dim, GetParam().neuron_dim}, 0.0);
    fill_random_uniform(som1.get_data_pointer(), som1.size());
    SOMType som2 = som1;

    auto&& f = GaussianFunctor(1.1, 0.2);

    MyTrainer_cpu trainer1(som1, f, 0, GetParam().num_rot, GetParam().use_flip, 0.0, Interpolation::BILINEAR, GetParam().neuron_dim);
    trainer1(data);

    MyTrainer_gpu trainer2(som2, f, 0, GetParam().num_rot, GetParam().use_flip, 0.0, Interpolation::BILINEAR, GetParam().neuron_dim, 256, DataType::FLOAT);
    trainer2(data);
    trainer2.update_som();

    EXPECT_EQ(som1.size(), som2.size());
    EXPECT_TRUE(EqualFloatArrays(som1.get_data_pointer(), som2.get_data_pointer(), som1.size(), 1e-4));
}

INSTANTIATE_TEST_CASE_P(TrainerCompareTest_all, compare_trainer_cpu,
    ::testing::Values(
        // som_dim, image_dim, neuron_dim, num_rot, use_flip
        TrainerCompareTestData(2, 2, 2, 1, false)
       ,TrainerCompareTestData(2, 2, 2, 4, false)
       ,TrainerCompareTestData(2, 2, 2, 8, false)
       ,TrainerCompareTestData(2, 2, 2, 1, true)
       ,TrainerCompareTestData(2, 2, 2, 4, true)
       ,TrainerCompareTestData(2, 2, 2, 8, true)

       ,TrainerCompareTestData(2, 4, 4, 1, false)
       ,TrainerCompareTestData(2, 2, 4, 1, false)

       ,TrainerCompareTestData(2, 64, 45, 360, true)
));
