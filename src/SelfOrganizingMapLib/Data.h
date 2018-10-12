/**
 * @file   SelfOrganizingMapLib/Data.h
 * @date   Oct 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stddef.h>
#include <array>
#include <functional>
#include <numeric>
#include <vector>

namespace pink {

//! Primary template for generic Data
template <typename DataLayout, typename T>
class Data
{
public:

    typedef T value_type;
    typedef Data<DataLayout, T> SelfType;
    typedef typename DataLayout::DimensionType DataDimensionType;

    /// Default construction
    Data()
     : data_dimension{0}
    {}

    /// Construction without initialization
    Data(DataDimensionType const& data_dimension)
     : data_dimension(data_dimension),
       data(get_size(data_dimension))
    {}

    /// Construction and initialize all element to value
    Data(DataDimensionType const& Data_dimension, T value)
     : data_dimension(data_dimension),
       data(get_size(data_dimension), value)
    {}

    /// Construction and copy data into Data
    Data(DataDimensionType const& data_dimension, T* data)
     : data_dimension(data_dimension),
       data(data, data + get_size(data_dimension))
    {}

    bool operator == (SelfType const& other) const
    {
        return data_dimension == other.data_dimension and
               data == other.data;
    }

    T* get_data_pointer() { return &data[0]; }
    T const* get_data_pointer() const { return &data[0]; }

    DataDimensionType get_dimension() const { return data_dimension; }

private:

    template <typename T2, size_t dim>
    T2 get_size(std::array<T2, dim> const& dimension) {
        return std::accumulate(dimension.begin(), dimension.end(), 1, std::multiplies<T2>());
    }

    DataDimensionType data_dimension;

    std::vector<T> data;

};

} // namespace pink
