/*
 * main.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Bernd Doser, HITS gGmbH
 */

#include "gtest/gtest.h"
#include <string>

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
