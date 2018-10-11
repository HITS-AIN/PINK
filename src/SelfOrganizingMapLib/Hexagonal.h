/**
 * @file   SelfOrganizingMapLib/Hexagonal.h
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

struct HexagonalLayout
{
	static const size_t dimensionality = 1;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;
};

//! Hexagonal layout for SOM, only supported for 2-dimensional
template <typename T>
class Hexagonal
{
public:

    typedef T value_type;

    Hexagonal(uint32_t length)
     : length(length),
       data(get_size(length))
    {}

    T& get(uint32_t position)
    {
        return data[position];
    }

private:

    //! Return number of elements for a regular hexagon
    size_t get_size(uint32_t length) const
    {
        uint32_t radius = (length - 1)/2;
        return length * length - radius * (radius + 1);
    }

    uint32_t length;

    std::vector<T> data;

};

} // namespace pink
