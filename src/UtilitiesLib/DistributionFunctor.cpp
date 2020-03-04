/**
 * @file   UtilitiesLib/DistributionFunctor.cpp
 * @brief  Virtual functors for distribution functions used by updating SOM.
 * @date   Aug 7, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <stdexcept>

#include "DistributionFunctor.h"

namespace pink {

StepFunctor::StepFunctor(float value)
 : m_value(value)
{}

float StepFunctor::operator () (float distance) const
{
    if (distance <= m_value) return 1.0;
    return 0.0;
}

GaussianFunctor::GaussianFunctor(float sigma, float damping)
 : m_sigma(sigma),
   m_damping(damping)
{}

float GaussianFunctor::operator () (float distance) const
{
    return m_damping / (m_sigma * std::sqrt(2.0f * static_cast<float>(M_PI)))
                   * std::exp(-0.5f * std::pow((distance/m_sigma), 2.0f));
}

UnityGaussianFunctor::UnityGaussianFunctor(float sigma, float damping)
 : m_sigma(sigma),
   m_damping(damping)
{}

float UnityGaussianFunctor::operator () (float distance) const
{
    return m_damping * std::exp(-0.5f * std::pow((distance/m_sigma), 2.0f));
}


MexicanHatFunctor::MexicanHatFunctor(float sigma, float damping)
 : m_sigma(sigma),
   m_damping(damping)
{
    if (sigma <= 0.0f) throw std::runtime_error("MexicanHatFunctor: sigma <= 0 not defined.");
}

float MexicanHatFunctor::operator () (float distance) const
{
    float distance2 = distance * distance;
    float sigma2 = m_sigma * m_sigma;
    return 2.0f * m_damping / (std::sqrt(3.0f * m_sigma) * std::pow(static_cast<float>(M_PI), 0.25f))
                * (1.0f - distance2/sigma2) * std::exp(-distance2 / (2.0f * sigma2));
}

} // namespace pink
