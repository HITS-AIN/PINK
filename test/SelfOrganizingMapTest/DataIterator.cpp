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

void add_binary_section(std::stringstream& ss, std::vector<std::vector<float>> const& images)
{
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
    ss.write(reinterpret_cast<const char*>(&images[0][0]), width * height * sizeof(float));
    ss.write(reinterpret_cast<const char*>(&images[1][0]), width * height * sizeof(float));
}

TEST(DataIteratorTest, cartesian_2d_without_header)
{
    std::vector<std::vector<float>> images{{1, 2, 3, 4}, {5, 6, 7, 8}};

    std::stringstream ss;
    add_binary_section(ss, images);

    DataIterator<CartesianLayout<2>, float> iter(ss, 2ul);

    EXPECT_EQ(2UL, iter->get_dimension()[0]);
    EXPECT_EQ(2UL, iter->get_dimension()[1]);

    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[0])), *iter);
    ++iter;
    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[1])), *iter);
    ++iter;
    EXPECT_EQ((DataIterator<CartesianLayout<2>, float>(ss, true)), iter);
}

TEST(DataIteratorTest, cartesian_2d_with_header)
{
    std::vector<std::vector<float>> images{{1, 2, 3, 4}, {5, 6, 7, 8}};

    std::stringstream ss;
    ss << "# test header\n";
    ss << "# END OF HEADER\n";
    add_binary_section(ss, images);

    DataIterator<CartesianLayout<2>, float> iter(ss, 2ul);

    EXPECT_EQ(2UL, iter->get_dimension()[0]);
    EXPECT_EQ(2UL, iter->get_dimension()[1]);

    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[0])), *iter);
    ++iter;
    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[1])), *iter);
    ++iter;
    EXPECT_EQ((DataIterator<CartesianLayout<2>, float>(ss, true)), iter);
}
