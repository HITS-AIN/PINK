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

/// Abstract base class
struct DataBase
{
    virtual ~DataBase() {}
};

/// Primary template for generic Data
template <typename Layout, typename T>
class Data : public DataBase
{
public:

    typedef T ValueType;
    typedef Data<Layout, T> SelfType;
    typedef Layout LayoutType;
    typedef typename LayoutType::DimensionType DimensionType;

    /// Default construction
    Data()
     : m_layout()
    {}

    /// Construction without initialization
    explicit Data(LayoutType const& layout)
     : m_layout(layout),
       m_data(layout.size())
    {}

    /// Construction and initialize all elements to value
    Data(LayoutType const& layout, T value)
     : m_layout(layout),
       m_data(layout.size(), value)
    {}

    /// Construction and copy data
    Data(LayoutType const& layout, std::vector<T> const& data)
     : m_layout(layout),
       m_data(data)
    {}

    /// Construction and move data
    Data(LayoutType const& layout, std::vector<T>&& data)
     : m_layout(layout),
       m_data(data)
    {}

    /// Copy construction
    template <typename T2>
    Data(Data<Layout, T2> const& other)
     : m_layout(other.layout),
       m_data(other.data.begin(), other.data.end())
    {}

    /// Copy assignment
    template <typename T2>
    Data& operator = (Data<Layout, T2> const& other)
    {
        std::swap(m_layout, other.layout);
        std::swap(m_data, other.data);
        return *this;
    }

    auto operator == (SelfType const& other) const
    {
        return m_layout == other.m_layout and
               m_data == other.m_data;
    }

    auto size() const { return m_data.size(); }

    auto get_data() { return m_data; }
    auto get_data() const { return m_data; }

    /// Return the element
    auto operator [] (uint32_t position) -> T& { return m_data[position]; }
    auto operator [] (uint32_t position) const -> T const& { return m_data[position]; }

    auto operator [] (DimensionType const& position) -> T& { return m_data[m_layout.get_index(position)]; }
    auto operator [] (DimensionType const& position) const -> T const& { return m_data[m_layout.get_index(position)]; }

    auto get_data_pointer() { return &m_data[0]; }
    auto get_data_pointer() const { return &m_data[0]; }

    auto get_layout() -> LayoutType { return m_layout; }
    auto get_layout() const -> LayoutType const { return m_layout; }

    auto get_dimension() -> DimensionType { return m_layout.m_dimension; }
    auto get_dimension() const -> DimensionType const { return m_layout.m_dimension; }

private:

    template <typename, typename>
    friend class Data;

    template <typename A, typename B>
    friend void write(Data<A, B> const& data, std::string const& filename);

    LayoutType m_layout;

    std::vector<T> m_data;

};

} // namespace pink
