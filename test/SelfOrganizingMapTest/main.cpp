/**
 * @file   ImageProcessingTest/main.cpp
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <omp.h>
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(1);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
