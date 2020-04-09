/**
 * @file   SelfOrganizingMapTest/circular_ed.cpp
 * @date   Apr 9, 2020
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
#include "SelfOrganizingMapLib/Mapper.h"
#include "UtilitiesLib/DistributionFunctor.h"

using namespace pink;

TEST(SelfOrganizingMapTest, circular_ed)
{
    typedef Data<CartesianLayout<2>, float> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Mapper<CartesianLayout<2>, CartesianLayout<2>, float, false> MapperType;

    auto image_dim = 4U;
    auto image_vec = std::vector<float>(image_dim * image_dim, 0.0);
    image_vec[4] = 1.0;

    auto som_dim = 2U;
    auto neuron_dim = 4U;
    auto som_vec = std::vector<float>(som_dim * som_dim * neuron_dim * neuron_dim, 0.0);
    som_vec[4] = 1.0; // [0, 0, 1, 0]
    som_vec[18] = 0.5; // [0, 1, 0, 2]

    DataType image({image_dim, image_dim}, image_vec);
    SOMType som({som_dim, som_dim}, {neuron_dim, neuron_dim}, som_vec);

    auto num_rot = 4;
    auto flip = false;
    auto euclidean_distance_dim = 4;

    MapperType mapper(som, 0, num_rot, flip, Interpolation::BILINEAR, euclidean_distance_dim, EuclideanDistanceShape::CIRCULAR);
    auto result = mapper(image);

    EXPECT_EQ((std::vector<float>{0.0, 0.5, 1.0, 1.0}), std::get<0>(result));
    EXPECT_EQ((std::vector<uint32_t>{0, 1, 0, 0}), std::get<1>(result));
}
