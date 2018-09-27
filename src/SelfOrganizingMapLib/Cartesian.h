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
#include <string>
#include <vector>

namespace pink {

template <class T>
struct Info;

//! n-dimensional cartesian layout
template <uint8_t dim, typename T>
class Cartesian
{
public:

    typedef T value_type;
	typedef Cartesian<dim, T> SelfType;
	typedef typename std::array<uint32_t, dim> DimensionType;

    /// Default construction
    Cartesian()
     : dimension{0}
    {}

    /// Construction without initialization
    Cartesian(std::array<uint32_t, dim> length)
     : dimension(length),
	   data(std::accumulate(length.begin(), length.end(), 1, std::multiplies<uint32_t>()))
    {}

    /// Construction and copy data into SOM
    Cartesian(DimensionType const& dimension, T* data)
     : dimension(dimension),
	   data(data, data +
           std::accumulate(dimension.begin(), dimension.end(), 1, std::multiplies<uint32_t>()))
    {}

    /// Construction and copy vector
    Cartesian(DimensionType const& dimension, std::vector<T> const& data)
     : dimension(dimension),
	   data(data)
    {}

    /// Construction and move vector
    Cartesian(DimensionType const& dimension, std::vector<T>&& data)
     : dimension(dimension),
	   data(data)
    {}

    bool operator == (SelfType const& other) const
	{
		return dimension == other.dimension and data == other.data;
	}

    T* get_data_pointer() { return &data[0]; }
    T const* get_data_pointer() const { return &data[0]; }

    T& get(DimensionType const& position)
    {
    	size_t p = 0;
    	for (uint8_t i = 0; i != dim; ++i) p += position[i] * i;
        return data[p];
    }

    T const& get(DimensionType const& position) const
    {
    	size_t p = 0;
    	for (uint8_t i = 0; i != dim; ++i) p += position[i] * i;
        return data[p];
    }

    std::array<uint32_t, dim> get_dimension() const { return dimension; }

    std::string info() const
    {
    	return std::string("Cartesian<") + std::to_string(dim) + ", " + Info<T>::name() + ">";
    }

private:

    DimensionType dimension;

    std::vector<T> data;

};

inline std::ostream& operator << (std::ostream& os, Cartesian<2, float> const& cartesian)
{
	auto&& dimension = cartesian.get_dimension();
	auto&& data = cartesian.get_data_pointer();
	for (size_t i = 0; i != dimension[0] * dimension[1]; ++i) os << data[i] << " ";
    return os << std::endl;
}

template <>
struct Info<float>
{
	static const std::string name() { return "float"; }
};

template <uint8_t dim, typename T>
struct Info<Cartesian<dim, T>>
{
	static const std::string name() { return std::string("Cartesian<") + std::to_string(dim) + ", " + Info<T>::name() + ">"; }
};

} // namespace pink
