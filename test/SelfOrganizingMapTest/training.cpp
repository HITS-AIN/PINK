/**
 * @file   SelfOrganizingMapTest/training.cpp
 * @brief  Unit tests for image processing.
 * @date   Oct 6, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <algorithm>
#include <cmath>
#include "gtest/gtest.h"
#include <vector>

#include "SelfOrganizingMapLib/SOM.h"

using namespace pink;

TEST(SelfOrganizingMapTest, quadratic)
{
	InputData input_data;
	SOM som(input_data);
}
