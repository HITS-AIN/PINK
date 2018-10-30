/**
 * @file   SelfOrganizingMapTest/Trainer.cpp
 * @brief  Unit tests for image processing.
 * @date   Sep 17, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <cmath>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "SelfOrganizingMapLib/Trainer.h"
#include "UtilitiesLib/DistributionFunctor.h"

#include "gtest/gtest.h"

using namespace pink;

TEST(SelfOrganizingMapTest, trainer_num_rot)
{
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> MySOM;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false> MyTrainer;

    uint32_t som_dim = 2;
    uint32_t neuron_dim = 2;
    MySOM som({som_dim, som_dim}, {neuron_dim, neuron_dim}, 0.0);

    auto&& f = GaussianFunctor(1.1, 0.2);

    EXPECT_THROW(MyTrainer(som, f, 0,  0, false, neuron_dim, 1.0, Interpolation::BILINEAR), pink::exception);
    EXPECT_THROW(MyTrainer(som, f, 0,  3, false, neuron_dim, 1.0, Interpolation::BILINEAR), pink::exception);
    EXPECT_THROW(MyTrainer(som, f, 0, 90, false, neuron_dim, 1.0, Interpolation::BILINEAR), pink::exception);

    EXPECT_NO_THROW(MyTrainer(som, f, 0,   1, false, neuron_dim, 1.0, Interpolation::BILINEAR));
    EXPECT_NO_THROW(MyTrainer(som, f, 0,   4, false, neuron_dim, 1.0, Interpolation::BILINEAR));
    EXPECT_NO_THROW(MyTrainer(som, f, 0, 720, false, neuron_dim, 1.0, Interpolation::BILINEAR));
}

TEST(SelfOrganizingMapTest, DISABLED_trainer_cartesian_2d)
{
    typedef Data<CartesianLayout<2>, uint8_t> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, uint8_t> SOMType;
    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, uint8_t, false> MyTrainer;

    uint32_t som_dim = 2;
    uint32_t image_dim = 2;
    uint32_t neuron_dim = 2;

    DataType image({image_dim, image_dim}, {1, 1, 1, 1});
    SOMType som({som_dim, som_dim}, {neuron_dim, neuron_dim}, 0);

    auto&& f = GaussianFunctor(1.1, 0.2);

    MyTrainer trainer(som, f, 0, 4, false, neuron_dim, 1.0, Interpolation::BILINEAR);
    trainer(image);

    auto&& v1 = f(0.0);
    auto&& v2 = f(1.0);
    auto&& v3 = f(std::sqrt(2.0));

    for (int i = 0; i != 4; ++i) {
        EXPECT_FLOAT_EQ(v1, som.get_neuron({0, 0})[i]);
        EXPECT_FLOAT_EQ(v2, som.get_neuron({1, 0})[i]);
        EXPECT_FLOAT_EQ(v2, som.get_neuron({0, 1})[i]);
        EXPECT_FLOAT_EQ(v3, som.get_neuron({1, 1})[i]);
    }
}
