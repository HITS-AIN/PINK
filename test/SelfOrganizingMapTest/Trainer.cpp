/**
 * @file   SelfOrganizingMapTest/Trainer.cpp
 * @date   Sep 17, 2018
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
#include "UtilitiesLib/DistributionFunctor.h"

using namespace pink;

TEST(SelfOrganizingMapTest, trainer_num_rot)
{
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> MySOM;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false> MyTrainer;

    uint32_t som_dim = 2;
    uint32_t neuron_dim = 2;
    uint32_t euclidean_distance_dim = 2;
    MySOM som({som_dim, som_dim}, {neuron_dim, neuron_dim}, 0.0);

    auto&& f = GaussianFunctor(1.1f, 0.2f);

    EXPECT_THROW(MyTrainer(som, f, 0,  0, false, 1.0,
        Interpolation::BILINEAR, euclidean_distance_dim), pink::exception);
    EXPECT_THROW(MyTrainer(som, f, 0,  3, false, 1.0,
        Interpolation::BILINEAR, euclidean_distance_dim), pink::exception);
    EXPECT_THROW(MyTrainer(som, f, 0, 90, false, 1.0,
        Interpolation::BILINEAR, euclidean_distance_dim), pink::exception);

    EXPECT_NO_THROW(MyTrainer(som, f, 0,   1, false, 1.0,
        Interpolation::BILINEAR, euclidean_distance_dim));
    EXPECT_NO_THROW(MyTrainer(som, f, 0,   4, false, 1.0,
        Interpolation::BILINEAR, euclidean_distance_dim));
    EXPECT_NO_THROW(MyTrainer(som, f, 0, 720, false, 1.0,
        Interpolation::BILINEAR, euclidean_distance_dim));
}

TEST(SelfOrganizingMapTest, trainer_cartesian_2d_int)
{
    typedef Data<CartesianLayout<2>, int> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, int> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, int, false> MyTrainer;

    uint32_t som_dim = 2;
    uint32_t image_dim = 2;
    uint32_t neuron_dim = 2;
    uint32_t euclidean_distance_dim = 2;

    DataType data({image_dim, image_dim}, {1000, 2000, 3000, 4000});
    SOMType som({som_dim, som_dim}, {neuron_dim, neuron_dim}, std::vector<int>(16, 0));

    auto&& f = GaussianFunctor(1.1f, 0.2f);

    MyTrainer trainer(som, f, 0, 1, false, 0.0, Interpolation::BILINEAR, euclidean_distance_dim);
    trainer(data);

    EXPECT_EQ(290, (som.get_neuron({0, 0}) [{1, 1}] ));
}

TEST(SelfOrganizingMapTest, trainer_cartesian_2d_float)
{
    typedef Data<CartesianLayout<2>, float> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false> MyTrainer;

    uint32_t som_dim = 1;
    uint32_t image_dim = 4;
    uint32_t neuron_dim = 2;
    uint32_t euclidean_distance_dim = 2;

    DataType data({image_dim, image_dim},
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    SOMType som({som_dim, som_dim}, {neuron_dim, neuron_dim}, std::vector<float>(4, 0.0));

    auto&& f = StepFunctor(10.0f);

    MyTrainer trainer(som, f, 0, 1, false, 0.0, Interpolation::BILINEAR, euclidean_distance_dim);
    trainer(data);

    DataType expected{{neuron_dim, neuron_dim}, {6.0, 7.0, 10.0, 11.0}};
    auto actual = som.get_neuron({0, 0});
    EXPECT_EQ(expected, actual);
}

TEST(SelfOrganizingMapTest, trainer_cartesian_3d_float)
{
    typedef Data<CartesianLayout<3>, float> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<3>, float> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<3>, float, false> MyTrainer;

    CartesianLayout<2> som_dim{2, 2};
    CartesianLayout<3> neuron_dim{2, 2, 2};
    auto data_dim = neuron_dim;
    uint32_t euclidean_distance_dim = 2;

    std::vector<float> raw_data(8);
    std::iota(raw_data.begin(), raw_data.end(), 1.0);

    DataType data(data_dim, raw_data);

    std::vector<float> raw_som(32, 0.0);
    std::vector<float> rot1{{4, 3, 2, 1, 8, 7, 6, 5}};
    std::copy_n(rot1.begin(), 8, &raw_som[8]);
    SOMType som(som_dim, neuron_dim, raw_som);

    auto&& f = StepFunctor(0.0f);

    MyTrainer trainer(som, f, 0, 4, false, 0.0, Interpolation::BILINEAR, euclidean_distance_dim);
    trainer(data);

    DataType expected{neuron_dim, rot1};
    auto actual = som.get_neuron({0, 1});
    EXPECT_EQ(expected, actual);
}
