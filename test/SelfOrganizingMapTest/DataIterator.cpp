/**
 * @file   SelfOrganizingMapTest/DataIterator.cpp
 * @date   Dec 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/DataIterator.h"

using namespace pink;

TEST(DataIteratorTest, cartesian_2d)
{
    std::vector<float> pixels1{1, 2, 3, 4};
    std::vector<float> pixels2{1, 2, 3, 4};

    std::stringstream ss;
    int version = 2;
    int binary_file_type = 0;
    int data_type = 0;
    int number_of_data_entries = 2;
    int layout = 0;
    int dimensionality = 2;
    int width = 2;
    int height = 2;
    ss.write(reinterpret_cast<const char*>(&version), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&binary_file_type), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&data_type), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&number_of_data_entries), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&layout), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&dimensionality), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&width), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&height), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&pixels1[0]), width * height * sizeof(float));
    ss.write(reinterpret_cast<const char*>(&pixels2[0]), width * height * sizeof(float));

    DataIterator<CartesianLayout<2>, float> iter(ss);

    EXPECT_EQ(2UL, iter->get_dimension()[0]);
    EXPECT_EQ(2UL, iter->get_dimension()[1]);

    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, pixels1)), *iter);
    ++iter;
    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, pixels2)), *iter);
    ++iter;
    EXPECT_EQ((DataIterator<CartesianLayout<2>, float>(ss, true)), iter);
}
