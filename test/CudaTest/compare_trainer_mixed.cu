/**
 * @file   CudaTest/compare_trainer_mixed.cpp
 * @brief  Compare generic GPU trainer with mixed precision generic GPU trainer.
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
    TrainerCompareTestData(uint32_t som_dim, uint32_t image_dim, uint32_t neuron_dim,
        uint32_t euclidean_distance_dim, uint32_t num_rot, bool use_flip)
     : m_som_dim(som_dim),
       m_image_dim(image_dim),
       m_neuron_dim(neuron_dim),
       m_euclidean_distance_dim(euclidean_distance_dim),
       m_num_rot(num_rot),
       m_use_flip(use_flip)
    {}

    uint32_t m_som_dim;
    uint32_t m_image_dim;
    uint32_t m_neuron_dim;
    uint32_t m_euclidean_distance_dim;

    uint32_t m_num_rot;
    bool m_use_flip;
};

class compare_trainer_mixed : public ::testing::TestWithParam<TrainerCompareTestData>
{};

TEST_P(compare_trainer_mixed, cartesian_2d_float)
{
    typedef Data<CartesianLayout<2>, float> DataContainerType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true> MyTrainer_gpu;

    DataContainerType data({GetParam().m_image_dim, GetParam().m_image_dim}, 0.0);
    fill_random_uniform(data.get_data_pointer(), data.size());

    SOMType som1({GetParam().m_som_dim, GetParam().m_som_dim},
        {GetParam().m_neuron_dim, GetParam().m_neuron_dim}, 0.0);
    fill_random_uniform(som1.get_data_pointer(), som1.size());
    SOMType som2 = som1;

    auto&& f = GaussianFunctor(1.1f, 0.2f);

    MyTrainer_gpu trainer1(som1, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, 0.0,
        Interpolation::BILINEAR, GetParam().m_euclidean_distance_dim, 256, DataType::FLOAT);
    trainer1(data);
    trainer1.update_som();

    MyTrainer_gpu trainer2(som2, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, 0.0,
        Interpolation::BILINEAR, GetParam().m_euclidean_distance_dim, 256, DataType::UINT8);
    trainer2(data);
    trainer2.update_som();

    EXPECT_EQ(som1.size(), som2.size());
    EXPECT_TRUE(EqualFloatArrays(som1.get_data_pointer(), som2.get_data_pointer(), som1.size(), 1e-4f));
}

INSTANTIATE_TEST_CASE_P(TrainerCompareTest_all, compare_trainer_mixed,
    ::testing::Values(
        // som_dim, image_dim, neuron_dim, euclidean_distance_dim, num_rot, use_flip
        TrainerCompareTestData(2,  2,   2,   2,   1, false)
       ,TrainerCompareTestData(2,  2,   2,   2,   4, false)
       ,TrainerCompareTestData(2,  2,   2,   2,   8, false)
       ,TrainerCompareTestData(2,  2,   2,   2,   1,  true)
       ,TrainerCompareTestData(2,  2,   2,   2,   4,  true)
       ,TrainerCompareTestData(2,  2,   2,   2,   8,  true)
       ,TrainerCompareTestData(2,  4,   4,   4,   1, false)
       ,TrainerCompareTestData(2,  2,   4,   4,   1, false)
       ,TrainerCompareTestData(2, 64,  45,  45, 360,  true)
       ,TrainerCompareTestData(2, 64, 100, 100, 360,  true)
       ,TrainerCompareTestData(2, 64, 100,  45, 360,  true)
));
