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
	MapperTestData(int som_dim, int neuron_dim, int image_dim, int euclidean_distance_dim, int num_rot, bool flip,
		float neuron_value, float image_value, std::vector<float> result)
      : som_dim(som_dim),
		neuron_dim(neuron_dim),
		image_dim(image_dim),
		euclidean_distance_dim(euclidean_distance_dim),
		num_rot(num_rot),
		flip(flip),
		neuron_value(neuron_value),
		image_value(image_value),
		result(result),
        som_size(som_dim * som_dim),
        neuron_size(neuron_dim * neuron_dim),
		image_size(image_dim * image_dim),
        som_total_size(som_size * neuron_size)
    {}

    int som_dim;
    int neuron_dim;
    int image_dim;
    int euclidean_distance_dim;
    int num_rot;
    bool flip;
    std::vector<float> result;
    float neuron_value;
    float image_value;

    int som_size;
    int neuron_size;
    int image_size;
    int som_total_size;
};

class MapperTest : public ::testing::TestWithParam<MapperTestData>
{};


TEST_P(MapperTest, mapper_cartesian_2d_float)
{
    typedef Data<CartesianLayout<2>, float> DataType;
    typedef SOM<CartesianLayout<2>, CartesianLayout<2>, float> SOMType;
    typedef Mapper<CartesianLayout<2>, CartesianLayout<2>, float, false> MapperType;

    auto data = GetParam();

    DataType image({data.image_dim, data.image_dim}, std::vector<float>(data.image_size, data.image_value));
    SOMType som({data.som_dim, data.som_dim}, {data.neuron_dim, data.neuron_dim}, std::vector<float>(data.som_total_size, data.neuron_value));

    MapperType mapper(som, 0, data.num_rot, data.flip, Interpolation::BILINEAR, data.euclidean_distance_dim);
    auto result = mapper(image);

    EXPECT_EQ(data.result, std::get<0>(result));
}

INSTANTIATE_TEST_CASE_P(MapperTest_all, MapperTest,
    ::testing::Values(
    	// som_dim, neuron_dim, image_dim, euclidean_distance_dim, num_rot, flip, neuron_value, image_value, result
        MapperTestData(1, 2, 2, 2,   1, false, 0.0, 1.0, {2.0}),
        MapperTestData(1, 3, 3, 2,   1, false, 0.0, 1.0, {2.0}),
        MapperTestData(1, 3, 5, 2, 360, false, 0.0, 1.0, {2.0}),
        MapperTestData(2, 3, 5, 2, 360, false, 0.0, 1.0, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(2, 3, 5, 2, 360,  true, 0.0, 1.0, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(2, 3, 5, 2, 360,  true, 0.0, 1.0, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(1, 2, 2, 2,   1, false, 1.0, 0.0, {2.0}),
        MapperTestData(1, 3, 3, 2,   1, false, 1.0, 0.0, {2.0}),
        MapperTestData(1, 3, 5, 2, 360, false, 1.0, 0.0, {2.0}),
        MapperTestData(2, 3, 5, 2, 360, false, 1.0, 0.0, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(2, 3, 5, 2, 360,  true, 1.0, 0.0, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(2, 3, 5, 2, 360,  true, 1.0, 0.0, {2.0, 2.0, 2.0, 2.0}),
        MapperTestData(1, 2, 2, 2,   1, false, 0.0, 0.5, {1.0}),
        MapperTestData(1, 2, 2, 2,   1, false, 0.5, 0.0, {1.0}),
        MapperTestData(1, 2, 2, 2,   1, false, 0.5, 0.5, {0.0}),
        MapperTestData(2, 2, 2, 2,   1, false, 0.0, 0.5, {1.0, 1.0, 1.0, 1.0}),
        MapperTestData(2, 2, 2, 2,   1, false, 0.5, 0.0, {1.0, 1.0, 1.0, 1.0})
));
