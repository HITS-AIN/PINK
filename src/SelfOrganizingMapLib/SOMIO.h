/**
 * @file   SelfOrganizingMapLib/SOMIO.h
 * @date   Oct 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iomanip>
#include <iostream>

#include "SelfOrganizingMapLib/SOM.h"

namespace pink {

template <typename SOMLayout, typename NeuronLayout, typename T>
std::ostream& operator << (std::ostream& os, SOM<SOMLayout, NeuronLayout, T> const& som)
{
    for (auto&& e : som.m_data) os << e << " ";
    return os << std::endl;
}

template <typename SOMLayout, typename NeuronLayout>
std::ostream& operator << (std::ostream& os, SOM<SOMLayout, NeuronLayout, uint8_t> const& som)
{
    for (auto&& e : som.m_data) os << static_cast<int>(e) << " ";
    return os << std::endl;
}

} // namespace pink
