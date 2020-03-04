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
    virtual ~DistributionFunctorBase() = default;

    virtual float operator () (float distance) const = 0;
};

/**
 * @brief Functor for step function
 *
 * return 1.0 if distance <= value, else 0.0
 */
struct StepFunctor : public DistributionFunctorBase
{
    explicit StepFunctor(float value);

    float operator () (float distance) const;

private:

    float m_value;
};

/**
 * @brief Functor for gaussian
 *
 * 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-1.0/2.0 * (x / sigma)**2 )
 */
struct GaussianFunctor : public DistributionFunctorBase
{
    GaussianFunctor(float sigma, float damping);

    float operator () (float distance) const;

private:

    float m_sigma;
    float m_damping;
};

/**
 * @brief Functor for gaussian normalised to unity
 *
 * 1.0 * math.exp(-1.0/2.0 * (x / sigma)**2 )
 */
struct UnityGaussianFunctor : public DistributionFunctorBase
{
    UnityGaussianFunctor(float sigma, float damping);

    float operator () (float distance) const;

private:

    float m_sigma;
    float m_damping;
};

/**
 * @brief Functor for mexican hat.
 *
 * 2.0 / ( math.sqrt(3.0 * sigma) * math.pow(math.pi, 0.25))
 *     * (1 - x**2.0 / sigma**2.0) * math.exp(-x**2.0/(2.0 * sigma**2))
 */
struct MexicanHatFunctor : public DistributionFunctorBase
{
    MexicanHatFunctor(float sigma, float damping);

    float operator () (float distance) const;

private:

    float m_sigma;
    float m_damping;

};

} // namespace pink
