/**
 * @file   UtilitiesLib/DistributionFunctions.cpp
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "DistributionFunctions.h"
#include <cmath>

float mexicanHat(float x, float sigma)
{
	float x2 = x * x;
	float sigma2 = sigma * sigma;
    return 2.0 / (sqrt(3.0 * sigma) * pow(M_PI, 0.25)) * (1.0 - x2/sigma2) * exp(-x2 / (2.0 * sigma2));
}

float gaussian(float x, float sigma)
{
    return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * pow((x/sigma),2));
}
