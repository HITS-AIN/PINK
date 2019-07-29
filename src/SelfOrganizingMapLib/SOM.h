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

/// Generic SOM
template <typename SOMLayout, typename NeuronLayout, typename T>
class SOM
{
public:

    typedef T ValueType;
    typedef SOMLayout SOMLayoutType;
    typedef NeuronLayout NeuronLayoutType;
    typedef SOM<SOMLayout, NeuronLayout, T> SelfType;
    typedef Data<NeuronLayout, T> NeuronType;

    /// Default construction
    SOM()
     : som_layout{0},
       neuron_layout{0}
    {}

    /// Construction by input data
    SOM(InputData const& input_data)
     : som_layout{extract_layout<SOMLayout::dimensionality>(input_data.som_width,
       input_data.som_height, input_data.som_depth)},
       neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
       data(som_layout.size() * neuron_layout.size())
    {
        // Initialize SOM
        if (input_data.init == SOMInitialization::ZERO)
            fill_value(&data[0], data.size());
        else if (input_data.init == SOMInitialization::RANDOM)
            fill_random_uniform(&data[0], data.size(), input_data.seed);
        else if (input_data.init == SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION) {
            fill_random_uniform(&data[0], data.size(), input_data.seed);
            for (int n = 0; n < input_data.som_size; ++n)
                for (uint32_t i = 0; i < input_data.neuron_dim; ++i)
                    data[n * input_data.neuron_size + i * input_data.neuron_dim + i] = 1.0;
        }
        else if (input_data.init == SOMInitialization::FILEINIT) {
            std::ifstream is(input_data.som_filename);
            if (!is) throw pink::exception("Error opening " + input_data.som_filename);

            header = get_file_header(is);

            // Ignore first three entries
            is.seekg((9 + SOMLayout::dimensionality) * sizeof(int), is.cur);
            is.read((char*)&data[0], data.size() * sizeof(float));
        } else
            throw pink::exception("Unknown SOMInitialization");
    }

    /// Construction without initialization
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout)
     : som_layout(som_layout),
       neuron_layout(neuron_layout),
       data(som_layout.size() * neuron_layout.size())
    {}

    /// Construction and initialize all elements to value
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout, T value)
     : som_layout(som_layout),
       neuron_layout(neuron_layout),
       data(som_layout.size() * neuron_layout.size(), value)
    {}

    /// Construction and copy data
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout,
        std::vector<T> const& data)
     : som_layout(som_layout),
       neuron_layout(neuron_layout),
       data(data)
    {}

    /// Construction and move data
    SOM(SOMLayoutType const& som_layout, NeuronLayoutType const& neuron_layout,
        std::vector<T>&& data)
     : som_layout(som_layout),
       neuron_layout(neuron_layout),
       data(data)
    {}

    auto operator == (SelfType const& other) const
    {
        return som_layout == other.som_layout and
               neuron_layout == other.neuron_layout and
               data == other.data;
    }

    auto size() const { return data.size(); }

    auto get_data() { return data; }
    auto get_data() const { return data; }

    auto get_data_pointer() { return &data[0]; }
    auto get_data_pointer() const { return &data[0]; }

    auto get_neuron(SOMLayoutType const& position) {
        auto&& beg = data.begin()
                   + (position.dimension[0] * som_layout.dimension[1]
                   + position.dimension[1]) * neuron_layout.size();
        auto&& end = beg + neuron_layout.size();
        return NeuronType(neuron_layout, std::vector<T>(beg, end));
    }

    auto get_number_of_neurons() const -> uint32_t { return som_layout.size(); }
    auto get_neuron_size() const -> uint32_t { return neuron_layout.size(); }

    auto get_som_layout() -> SOMLayoutType { return som_layout; }
    auto get_som_layout() const -> SOMLayoutType const { return som_layout; }
    auto get_neuron_layout() -> NeuronLayoutType { return neuron_layout; }
    auto get_neuron_layout() const -> NeuronLayoutType const { return neuron_layout; }

    auto get_som_dimension() -> typename SOMLayoutType::DimensionType {
        return som_layout.dimension;
    }
    auto get_som_dimension() const -> typename SOMLayoutType::DimensionType const {
        return som_layout.dimension;
    }
    auto get_neuron_dimension() -> typename NeuronLayoutType::DimensionType {
        return neuron_layout.dimension;
    }
    auto get_neuron_dimension() const -> typename NeuronLayoutType::DimensionType const {
        return neuron_layout.dimension;
    }

private:

    template <typename A, typename B, typename C>
    friend void write(SOM<A, B, C> const& som, std::string const& filename);

    template <typename A, typename B, typename C>
    friend std::ostream& operator << (std::ostream& os, SOM<A, B, C> const& som);

    SOMLayoutType som_layout;

    NeuronLayoutType neuron_layout;

    // Header of initialization SOM, will be copied to resulting SOM
    std::string header;

    std::vector<T> data;

};

} // namespace pink
