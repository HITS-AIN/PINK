/**
 * @file   SelfOrganizingMapLib/Data.h
 * @date   Oct 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace pink {

//! Primary template for generic Data
template <typename Layout, typename T>
class Data
{
public:

    typedef T ValueType;
    typedef Data<Layout, T> SelfType;
    typedef Layout LayoutType;
    typedef typename LayoutType::DimensionType DimensionType;

    /// Default construction
    Data()
     : layout{0}
    {}

    /// Construction without initialization
    Data(LayoutType const& layout)
     : layout(layout),
       data(layout.size())
    {}

    /// Construction and initialize all elements to value
    Data(LayoutType const& layout, T value)
     : layout(layout),
       data(layout.size(), value)
    {}

    /// Construction and copy data
    Data(LayoutType const& layout, std::vector<T> const& data)
     : layout(layout),
       data(data)
    {}

    /// Construction and move data
    Data(LayoutType const& layout, std::vector<T>&& data)
     : layout(layout),
       data(data)
    {}

    /// Copy construction
    template <typename T2>
    Data(Data<Layout, T2> const& other)
     : layout(other.layout),
       data(other.data.begin(), other.data.end())
    {}

    auto operator == (SelfType const& other) const
    {
        return layout == other.layout and
               data == other.data;
    }

    auto size() const { return data.size(); }

    auto get_data() { return data; }
    auto get_data() const { return data; }

    /// Return the element
    auto operator [] (uint32_t position) -> T& { return data[position]; }
    auto operator [] (uint32_t position) const -> T const& { return data[position]; }

    auto operator [] (DimensionType const& position) -> T& { return data[layout.get_index(position)]; }
    auto operator [] (DimensionType const& position) const -> T const& { return data[layout.get_index(position)]; }

    auto get_data_pointer() { return &data[0]; }
    auto get_data_pointer() const { return &data[0]; }

    auto get_layout() -> LayoutType { return layout; }
    auto get_layout() const -> LayoutType const { return layout; }

    auto get_dimension() -> DimensionType { return layout.dimension; }
    auto get_dimension() const -> DimensionType const { return layout.dimension; }

private:

    template <typename, typename>
    friend class Data;

    template <typename A, typename B>
    friend void write(Data<A, B> const& data, std::string const& filename);

    LayoutType layout;

    std::vector<T> data;

};

} // namespace pink
