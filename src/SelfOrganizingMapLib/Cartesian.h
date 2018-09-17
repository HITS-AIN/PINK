/**
 * @file   SelfOrganizingMapLib/Cartesian.h
 * @date   Aug 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace pink {

//! n-dimensional cartesian layout for SOM
template <uint8_t dim, typename T>
class Cartesian
{
public:

    typedef T value_type;

    Cartesian(std::array<uint32_t, dim> length, T const& init_value)
     : length(length),
	   data(std::accumulate(length.begin(), length.end(), 1, std::multiplies<uint32_t>()), init_value)
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

private:

    std::array<uint32_t, dim> length;

    std::vector<T> data;

};

} // namespace pink
