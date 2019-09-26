/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <fstream>
#include <functional>
#include <vector>

#include "CartesianLayout.h"
#include "Data.h"
#include "HexagonalLayout.h"
#include "UtilitiesLib/Filler.h"
#include "UtilitiesLib/get_file_header.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/get_static_array.h"

namespace pink {

template <uint8_t dim>
inline std::array<uint32_t, dim> extract_layout(uint32_t x, uint32_t y, uint32_t z);

template <>
inline std::array<uint32_t, 1> extract_layout<1>(uint32_t x, uint32_t /*y*/, uint32_t /*z*/)
{
    return std::array<uint32_t, 1>{x};
}

template <>
inline std::array<uint32_t, 2> extract_layout<2>(uint32_t x, uint32_t y, uint32_t /*z*/)
{
    return std::array<uint32_t, 2>{x, y};
}

template <>
inline std::array<uint32_t, 3> extract_layout<3>(uint32_t x, uint32_t y, uint32_t z)
{
    return std::array<uint32_t, 3>{x, y, z};
}

/// Abstract base class
struct SOMBase
{
    virtual ~SOMBase() {}
};

/// Generic SOM
template <typename SOMLayout, typename NeuronLayout, typename T>
class SOM : public SOMBase
{
public:

    typedef T ValueType;
    typedef SOMLayout SOMLayoutType;
    typedef NeuronLayout NeuronLayoutType;
    typedef SOM<SOMLayout, NeuronLayout, T> SelfType;
    typedef Data<NeuronLayout, T> NeuronType;

    /// Default construction
    SOM()
     : m_som_layout{0},
       m_neuron_layout{0}
    {}

    /// Construction by input data
    explicit SOM(InputData const& input_data)
     : m_som_layout{extract_layout<SOMLayout::dimensionality>(input_data.m_som_width,
       input_data.m_som_height, input_data.m_som_depth)},
       m_neuron_layout{get_static_array<NeuronLayout::dimensionality>(input_data.m_neuron_dimension)},
       m_data(m_som_layout.size() * m_neuron_layout.size())
    {
        // Initialize SOM
        if (input_data.m_init == SOMInitialization::ZERO)
            fill_value(&m_data[0], m_data.size());
        else if (input_data.m_init == SOMInitialization::RANDOM)
            fill_random_uniform(&m_data[0], m_data.size(), input_data.m_seed);
        else if (input_data.m_init == SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION) {
            fill_random_uniform(&m_data[0], m_data.size(), input_data.m_seed);
            for (uint32_t n = 0; n < input_data.m_som_size; ++n)
                for (uint32_t i = 0; i < input_data.m_neuron_dim; ++i)
                    m_data[n * input_data.m_neuron_size + i * input_data.m_neuron_dim + i] = 1.0;
        }
        else if (input_data.m_init == SOMInitialization::FILEINIT) {
            std::ifstream is(input_data.m_som_filename);
            if (!is) throw pink::exception("Error opening " + input_data.m_som_filename);

            m_header = get_file_header(is);

            // Ignore first three entries
            is.seekg((9 + SOMLayout::dimensionality) * sizeof(int), is.cur);
            is.read(reinterpret_cast<char*>(&m_data[0]), static_cast<std::streamsize>(m_data.size() * sizeof(float)));
        } else
            throw pink::exception("Unknown SOMInitialization");
    }

    /// Construction without initialization
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout)
     : m_som_layout(som_layout),
       m_neuron_layout(neuron_layout),
       m_data(som_layout.size() * neuron_layout.size())
    {}

    /// Construction and initialize all elements to value
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout, T value)
     : m_som_layout(som_layout),
       m_neuron_layout(neuron_layout),
       m_data(som_layout.size() * neuron_layout.size(), value)
    {}

    /// Construction and copy data
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout,
        std::vector<T> const& data)
     : m_som_layout(som_layout),
       m_neuron_layout(neuron_layout),
       m_data(data)
    {}

    /// Construction and move data
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout,
        std::vector<T>&& data)
     : m_som_layout(som_layout),
       m_neuron_layout(neuron_layout),
       m_data(data)
    {}

    auto operator == (SelfType const& other) const
    {
        return m_som_layout == other.m_som_layout and
               m_neuron_layout == other.m_neuron_layout and
               m_data == other.m_data;
    }

    auto size() const { return m_data.size(); }

    auto get_data() { return m_data; }
    auto get_data() const { return m_data; }

    auto get_data_pointer() { return &m_data[0]; }
    auto get_data_pointer() const { return &m_data[0]; }

    auto get_neuron(SOMLayoutType const& position) {
        auto&& beg = m_data.begin()
                   + static_cast<int>((position.m_dimension[0] * m_som_layout.m_dimension[1]
                   + position.m_dimension[1]) * m_neuron_layout.size());
        auto&& end = beg + static_cast<int>(m_neuron_layout.size());
        return NeuronType(m_neuron_layout, std::vector<T>(beg, end));
    }

    auto get_number_of_neurons() const -> uint32_t { return static_cast<uint32_t>(m_som_layout.size()); }
    auto get_neuron_size() const -> uint32_t { return static_cast<uint32_t>(m_neuron_layout.size()); }

    auto get_som_layout() -> SOMLayoutType { return m_som_layout; }
    auto get_som_layout() const -> SOMLayoutType const { return m_som_layout; }
    auto get_neuron_layout() -> NeuronLayoutType { return m_neuron_layout; }
    auto get_neuron_layout() const -> NeuronLayoutType const { return m_neuron_layout; }

    auto get_som_dimension() -> typename SOMLayoutType::DimensionType {
        return m_som_layout.m_dimension;
    }
    auto get_som_dimension() const -> typename SOMLayoutType::DimensionType const {
        return m_som_layout.m_dimension;
    }
    auto get_neuron_dimension() -> typename NeuronLayoutType::DimensionType {
        return m_neuron_layout.m_dimension;
    }
    auto get_neuron_dimension() const -> typename NeuronLayoutType::DimensionType const {
        return m_neuron_layout.m_dimension;
    }

private:

    template <typename A, typename B, typename C>
    friend void write(SOM<A, B, C> const& som, std::string const& filename);

    template <typename A, typename B, typename C>
    friend std::ostream& operator << (std::ostream& os, SOM<A, B, C> const& som);

    SOMLayoutType m_som_layout;

    NeuronLayoutType m_neuron_layout;

    // Header of initialization SOM, will be copied to resulting SOM
    std::string m_header;

    std::vector<T> m_data;

};

} // namespace pink
