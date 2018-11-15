/**
 * @file   UtilitiesLib/DistributionFunctor.h
 * @brief  Virtual functors for distribution functions used by updating SOM.
 * @date   Dec 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <cmath>
#include <stdexcept>

namespace pink {

/**
 * @brief Abstract base for distribution functor.
 */
struct DistributionFunctorBase
{
    virtual float operator () (float distance) const = 0;

    virtual ~DistributionFunctorBase() {}
};

/**
 * @brief Functor for step function
 *
 * return 1.0 if distance <= value, else 0.0
 */
struct StepFunctor : public DistributionFunctorBase
{
    StepFunctor(float value)
     : value(value)
    {}

    float operator () (float distance) const
    {
        if (distance <= value) return 1.0;
        return 0.0;
    }

private:

    float value;
};

/**
 * @brief Functor for gaussian
 *
 * 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-1.0/2.0 * (x / sigma)**2 )
 */
struct GaussianFunctor : public DistributionFunctorBase
{
    GaussianFunctor(float sigma, float damping)
     : sigma(sigma),
       damping(damping)
    {}

    float operator () (float distance) const
    {
        return damping / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * pow((distance/sigma),2));
    }

private:

    float sigma;
    float damping;
};

/**
 * @brief Functor for mexican hat.
 *
 * 2.0 / ( math.sqrt(3.0 * sigma) * math.pow(math.pi, 0.25)) * (1- x**2.0 / sigma**2.0) * math.exp(-x**2.0/(2.0 * sigma**2))
 */
struct MexicanHatFunctor : public DistributionFunctorBase
{
    MexicanHatFunctor(float sigma, float damping)
     : sigma(sigma),
       damping(damping)
    {
        if (sigma <= 0.0) throw std::runtime_error("MexicanHatFunctor: sigma <= 0 not defined.");
    }

    float operator () (float distance) const
    {
        float distance2 = distance * distance;
        float sigma2 = sigma * sigma;
        return 2.0 * damping / (sqrt(3.0 * sigma) * pow(M_PI, 0.25)) * (1.0 - distance2/sigma2) * exp(-distance2 / (2.0 * sigma2));
    }

private:

    float sigma;
    float damping;
};

} // namespace pink
