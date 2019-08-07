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
    explicit StepFunctor(float value)
     : m_value(value)
    {}

    float operator () (float distance) const
    {
        if (distance <= m_value) return 1.0;
        return 0.0;
    }

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
    GaussianFunctor(float sigma, float damping)
     : m_sigma(sigma),
       m_damping(damping)
    {}

    float operator () (float distance) const
    {
        return m_damping / (m_sigma * std::sqrt(2.0f * static_cast<float>(M_PI)))
                       * std::exp(-0.5f * std::pow((distance/m_sigma), 2.0f));
    }

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
    MexicanHatFunctor(float sigma, float damping)
     : m_sigma(sigma),
       m_damping(damping)
    {
        if (sigma <= 0.0f) throw std::runtime_error("MexicanHatFunctor: sigma <= 0 not defined.");
    }

    float operator () (float distance) const
    {
        float distance2 = distance * distance;
        float sigma2 = m_sigma * m_sigma;
        return 2.0f * m_damping / (std::sqrt(3.0f * m_sigma) * std::pow(static_cast<float>(M_PI), 0.25f))
                    * (1.0f - distance2/sigma2) * std::exp(-distance2 / (2.0f * sigma2));
    }

private:

    float m_sigma;
    float m_damping;

};

} // namespace pink
