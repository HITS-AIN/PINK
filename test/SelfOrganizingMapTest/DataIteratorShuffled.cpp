/**
 * @file   SelfOrganizingMapTest/DataIteratorShuffled.cpp
 * @date   Dec 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>

#include "add_binary_section.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/DataIteratorShuffled.h"

using namespace pink;

TEST(DataIteratorShuffledTest, cartesian_2d_without_header)
{
    std::vector<std::vector<float>> images{{1, 2, 3, 4}, {5, 6, 7, 8}};

    std::stringstream ss;
    add_binary_section(ss, images);

    DataIteratorShuffled<CartesianLayout<2>, float> iter(ss, 2ul);

    EXPECT_EQ(2UL, iter->get_dimension()[0]);
    EXPECT_EQ(2UL, iter->get_dimension()[1]);

    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[0])), *iter);
    ++iter;
    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[1])), *iter);
    ++iter;
    EXPECT_EQ((DataIteratorShuffled<CartesianLayout<2>, float>(ss, true)), iter);
}

TEST(DataIteratorShuffledTest, cartesian_2d_with_header)
{
    std::vector<std::vector<float>> images{{1, 2, 3, 4}, {5, 6, 7, 8}};

    std::stringstream ss;
    ss << "# test header\n";
    ss << "# END OF HEADER\n";
    add_binary_section(ss, images);

    DataIteratorShuffled<CartesianLayout<2>, float> iter(ss, 2ul);

    EXPECT_EQ(2UL, iter->get_dimension()[0]);
    EXPECT_EQ(2UL, iter->get_dimension()[1]);

    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[0])), *iter);
    ++iter;
    EXPECT_EQ((Data<CartesianLayout<2>, float>({2, 2}, images[1])), *iter);
    ++iter;
    EXPECT_EQ((DataIteratorShuffled<CartesianLayout<2>, float>(ss, true)), iter);
}
