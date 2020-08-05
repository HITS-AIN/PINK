/**
 * @file   CudaTest/compare_trainer_3d_cpu.cpp
 * @brief  Compare generic GPU trainer against generic CPU trainer.
 * @date   Feb 3, 2020
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

struct TrainerCompare3DTestData
{
    TrainerCompare3DTestData(uint32_t som_dim, uint32_t image_dim, uint32_t image_depth, uint32_t neuron_dim,
        uint32_t euclidean_distance_dim, uint32_t num_rot, bool use_flip)
     : m_som_dim(som_dim),
       m_image_dim(image_dim),
       m_image_depth(image_depth),
       m_neuron_dim(neuron_dim),
       m_euclidean_distance_dim(euclidean_distance_dim),
       m_num_rot(num_rot),
       m_use_flip(use_flip)
    {}

    uint32_t m_som_dim;
    uint32_t m_image_dim;
    uint32_t m_image_depth;
    uint32_t m_neuron_dim;
    uint32_t m_euclidean_distance_dim;

    uint32_t m_num_rot;
    bool m_use_flip;
};

class compare_trainer_3d_cpu : public ::testing::TestWithParam<TrainerCompare3DTestData>
{};

TEST_P(compare_trainer_3d_cpu, cartesian_3d_float)
{
    typedef CartesianLayout<3> DataLayout;
    typedef CartesianLayout<2> SOMLayout;
    typedef Data<DataLayout, float> DataContainerType;
    typedef SOM<SOMLayout, DataLayout, float> SOMType;
    typedef Trainer<SOMLayout, DataLayout, float, false> MyTrainer_cpu;
    typedef Trainer<SOMLayout, DataLayout, float, true> MyTrainer_gpu;

    DataContainerType data({GetParam().m_image_depth, GetParam().m_image_dim, GetParam().m_image_dim}, 0.0);
    fill_random_uniform(data.get_data_pointer(), data.size());

    SOMType som1({GetParam().m_som_dim, GetParam().m_som_dim},
        {GetParam().m_image_depth, GetParam().m_neuron_dim, GetParam().m_neuron_dim}, 0.0);
    fill_random_uniform(som1.get_data_pointer(), som1.size());
    SOMType som2 = som1;

    auto&& f = GaussianFunctor(1.1f, 0.2f);

    MyTrainer_cpu trainer1(som1, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, -1.0,
        Interpolation::BILINEAR, GetParam().m_euclidean_distance_dim);
    trainer1(data);

    MyTrainer_gpu trainer2(som2, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, -1.0,
        Interpolation::BILINEAR, GetParam().m_euclidean_distance_dim, EuclideanDistanceShape::QUADRATIC, 256, DataType::FLOAT);
    trainer2(data);
    trainer2.update_som();

    EXPECT_EQ(som1.size(), som2.size());
    EXPECT_TRUE(EqualFloatArrays(som1.get_data_pointer(), som2.get_data_pointer(), som1.size(), 1e-4f));
}

INSTANTIATE_TEST_SUITE_P(TrainerCompare3DTest_all, compare_trainer_3d_cpu,
    ::testing::Values(
        // som_dim, image_dim, image_depth, neuron_dim, euclidean_distance_dim, num_rot, use_flip
        TrainerCompare3DTestData(2,   2,   2,   2,   2,   1, false)
       ,TrainerCompare3DTestData(2,   2,   2,   2,   2,   4, false)
       ,TrainerCompare3DTestData(2,   2,   2,   2,   2,   8, false)
       ,TrainerCompare3DTestData(2,   2,   2,   2,   2,   1,  true)
       ,TrainerCompare3DTestData(2,   2,   2,   2,   2,   4,  true)
       ,TrainerCompare3DTestData(2,   2,   2,   2,   2,   8,  true)
       ,TrainerCompare3DTestData(2,   4,   2,   4,   4,   1, false)
       ,TrainerCompare3DTestData(2,   2,   2,   4,   4,   1, false)
       ,TrainerCompare3DTestData(2,  64,   2,  45,  45, 360,  true)
       ,TrainerCompare3DTestData(2,  64,   2, 100, 100, 360,  true)
       ,TrainerCompare3DTestData(2,  64,   2, 100,  45, 360,  true)
       ,TrainerCompare3DTestData(2, 124,   2,  91,  64, 360,  true)
       ,TrainerCompare3DTestData(2,   4,   2,   2,   2,   1, false)
       ,TrainerCompare3DTestData(2,   4,   2,   2,   2,   8, false)
       ,TrainerCompare3DTestData(2,   4,   2,   2,   2,   1,  true)
       ,TrainerCompare3DTestData(2,   4,   2,   2,   2,   8,  true)
));
