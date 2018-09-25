/**
 * @file   SelfOrganizingMapLib/SOM_generic.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace pink {

//! Primary template for generic SOM
template <typename Layout, typename T>
class SOM_generic;

template <typename T>
class SOM_generic<Cartesian<3>, T>
{
public:

    typedef T value_type;

    /// Default construction
    SOM_generic()
     : data()
    {}

    T& get(std::array<uint32_t, dim> position)
    {
    	size_t p = 0;
    	for (uint8_t i = 0; i != dim; ++i) p += position[i] * i;
        return data[p];
    }

    T const& get(std::array<uint32_t, dim> position) const
    {
    	size_t p = 0;
    	for (uint8_t i = 0; i != dim; ++i) p += position[i] * i;
        return data[p];
    }

    std::array<uint32_t, dim> get_length() const { return length; }

private:

    std::array<uint32_t, dim> length;

    std::vector<T> data;

};

} // namespace pink
