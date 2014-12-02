/**
 * @file   UtilitiesLib/DistributionFunctions.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef DISTRIBUTIONFUNCTIONS_H_
#define DISTRIBUTIONFUNCTIONS_H_

#include <cmath>

#define UPDATE_NEURONS_SIGMA     1.1
#define UPDATE_NEURONS_DAMPING   0.2

struct DistributionFunctionBase
{
    virtual float operator ()(float distance) const = 0;
    virtual ~DistributionFunctionBase();
};

struct GaussianFunction : public DistributionFunctionBase
{
    GaussianFunction(float sigma) : sigma(sigma) {}

    float operator ()(float distance) const
    {
        return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * pow((distance/sigma),2));
    }

private:

    float sigma;

};

struct MexicanHatFunction : public DistributionFunctionBase
{
    MexicanHatFunction(float sigma) : sigma(sigma), sigma2(sigma*sigma) {}

    float operator ()(float distance) const
    {
        float distance2 = distance * distance;
        return 2.0 / (sqrt(3.0 * sigma) * pow(M_PI, 0.25)) * (1.0 - distance2/sigma2) * exp(-distance2 / (2.0 * sigma2));
    }

private:

    float sigma;
    float sigma2;

};

//! 2.0 / ( math.sqrt(3.0 * sigma) * math.pow(math.pi, 0.25)) * (1- x**2.0 / sigma**2.0) * math.exp(-x**2.0/(2.0 * sigma**2))
float mexicanHat(float x, float sigma);

//! 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-1.0/2.0 * (x / sigma)**2 );
float gaussian(float x, float sigma);

#endif /* DISTRIBUTIONFUNCTIONS_H_ */
