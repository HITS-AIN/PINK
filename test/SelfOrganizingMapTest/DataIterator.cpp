/**
 * @file   SelfOrganizingMapTest/DataIterator.cpp
 * @date   Dec 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <string>

#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/DataIterator.h"

using namespace pink;

TEST(DataIteratorTest, cartesian_2d)
{
    const std::string filename("image.bin");

    Data<CartesianLayout<2>, float> data({2, 2}, std::vector<float>({1, 2, 3, 4}));

    DataIterator<CartesianLayout<2>, float> iter(filename);

    EXPECT_EQ(*iter, data);
}
