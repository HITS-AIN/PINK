/**
 * @file   SelfOrganizingMapLib/Data.h
 * @date   Oct 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stddef.h>
#include <array>
#include <functional>
#include <vector>

namespace pink {

//! Primary template for generic Data
template <typename DataLayout, typename T>
class Data
{
public:

    typedef T ValueType;
    typedef Data<DataLayout, T> SelfType;
    typedef DataLayout DataLayoutType;

    /// Default construction
    Data()
     : data_dimension{0}
    {}

    /// Construction without initialization
    Data(DataLayoutType const& data_dimension)
     : data_dimension(data_dimension),
       data(data_dimension.get_size())
    {}

    /// Construction and initialize all element to value
    Data(DataLayoutType const& data_dimension, T value)
     : data_dimension(data_dimension),
       data(data_dimension.get_size(), value)
    {}

    /// Construction and copy data
    Data(DataLayoutType const& data_dimension, std::vector<T> const& data)
     : data_dimension(data_dimension),
       data(data)
    {}

    /// Construction and move data
    Data(DataLayoutType const& data_dimension, std::vector<T>&& data)
     : data_dimension(data_dimension),
       data(data)
    {}

    auto operator == (SelfType const& other) const
    {
        return data_dimension == other.data_dimension and
               data == other.data;
    }

    auto get_data() { return data; }
    auto get_data() const { return data; }

    auto get_data_pointer() { return &data[0]; }
    auto get_data_pointer() const { return &data[0]; }

    auto get_dimension() -> typename DataLayoutType::DimensionType { return data_dimension.dimension; }
    auto get_dimension() const -> typename DataLayoutType::DimensionType const { return data_dimension.dimension; }

private:

    template <typename A, typename B>
    friend void write(Data<A, B> const& data, std::string const& filename);

    DataLayoutType data_dimension;

    std::vector<T> data;

};

} // namespace pink
