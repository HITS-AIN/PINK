/**
 * @file   SelfOrganizingMapTest/Mapper.cpp
 * @date   Jul 10, 2019
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

struct MapperTestData
{
    MapperTestData(uint32_t som_dim, uint32_t neuron_dim, uint32_t image_dim, uint32_t euclidean_distance_dim,
        uint32_t num_rot, bool flip, std::vector<float> result)
      : som_dim(som_dim),
        neuron_dim(neuron_dim),
        image_dim(image_dim),
        euclidean_distance_dim(euclidean_distance_dim),
        num_rot(num_rot),
        flip(flip),
        result(result),
        som_size(som_dim * som_dim),
        neuron_size(neuron_dim * neuron_dim),
        image_size(image_dim * image_dim),
        som_total_size(som_size * neuron_size)
    {}

    uint32_t som_dim;
    uint32_t neuron_dim;
    uint32_t image_dim;
    uint32_t euclidean_distance_dim;
    uint32_t num_rot;
    bool flip;
    std::vector<float> result;
    uint32_t som_size;
    uint32_t neuron_size;
    uint32_t image_size;
    uint32_t som_total_size;
};

class MapperTest : public ::testing::TestWithParam<MapperTestData>
{};

TEST_P(MapperTest, mapper_cartesian_2d_float)
{
    typedef Data<CartesianLayout<2>, float> MyData;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Mapper<CartesianLayout<2>, CartesianLayout<2>, float, true> MapperType;

    auto data = GetParam();

    MyData image({data.image_dim, data.image_dim}, std::vector<float>(data.image_size, 0.0));
    SOMType som({data.som_dim, data.som_dim}, {data.neuron_dim, data.neuron_dim}, std::vector<float>(data.som_total_size, 1.0));

    MapperType mapper(som, 0, data.num_rot, data.flip, Interpolation::BILINEAR, data.euclidean_distance_dim, 256, DataType::FLOAT);
    auto result = mapper(image);

    EXPECT_EQ(data.result, std::get<0>(result));
}

TEST_P(MapperTest, mapper_cartesian_2d_uint8)
{
    typedef Data<CartesianLayout<2>, float> MyData;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Mapper<CartesianLayout<2>, CartesianLayout<2>, float, true> MapperType;

    auto data = GetParam();

    MyData image({data.image_dim, data.image_dim}, std::vector<float>(data.image_size, 0.0));
    SOMType som({data.som_dim, data.som_dim}, {data.neuron_dim, data.neuron_dim}, std::vector<float>(data.som_total_size, 1.0));

    MapperType mapper(som, 0, data.num_rot, data.flip, Interpolation::BILINEAR, data.euclidean_distance_dim, 256, DataType::UINT8);
    auto result = mapper(image);

    EXPECT_EQ(data.result, std::get<0>(result));
}

TEST_P(MapperTest, mapper_cartesian_2d_uint16)
{
    typedef Data<CartesianLayout<2>, float> MyData;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Mapper<CartesianLayout<2>, CartesianLayout<2>, float, true> MapperType;

    auto data = GetParam();

    MyData image({data.image_dim, data.image_dim}, std::vector<float>(data.image_size, 0.0));
    SOMType som({data.som_dim, data.som_dim}, {data.neuron_dim, data.neuron_dim}, std::vector<float>(data.som_total_size, 1.0));

    MapperType mapper(som, 0, data.num_rot, data.flip, Interpolation::BILINEAR, data.euclidean_distance_dim, 256, DataType::UINT16);
    auto result = mapper(image);

    EXPECT_EQ(data.result, std::get<0>(result));
}

INSTANTIATE_TEST_CASE_P(MapperTest_all, MapperTest,
    ::testing::Values(
        // som_dim, neuron_dim, image_dim, euclidean_distance_dim, num_rot, flip, result
        MapperTestData(1, 2, 2, 2,   1, false, {2.0}),
        MapperTestData(1, 3, 3, 2,   1, false, {2.0}),
        MapperTestData(1, 3, 3, 2, 360, false, {2.0}),
        MapperTestData(2, 3, 3, 2, 360, false, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(2, 3, 3, 2, 360,  true, {2.0, 2.0, 2.0, 2.0})
));
