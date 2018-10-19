/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <functional>
#include <stddef.h>
#include <thrust/device_vector.h>
#include <vector>

#include "CartesianLayout.h"
#include "Data.h"
#include "HexagonalLayout.h"
#include "UtilitiesLib/InputData.h"

namespace pink {

//! Primary template for generic SOM
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
     : som_dimension{0},
       neuron_dimension{0}
    {}

    /// Construction by input data
    SOM(InputData const& input_data);

    /// Construction without initialization
    SOM(SOMLayoutType const& som_dimension, NeuronLayoutType const& neuron_dimension)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(som_dimension.get_size() * neuron_dimension.get_size())
    {}

    /// Construction and initialize all element to value
    SOM(SOMLayoutType const& som_dimension, NeuronLayoutType const& neuron_dimension, T value)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(som_dimension.get_size() * neuron_dimension.get_size(), value)
    {}

    /// Construction and copy data into SOM
    SOM(SOMLayoutType const& som_dimension, NeuronLayoutType const& neuron_dimension, T* data)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(data, data + som_dimension.get_size() * neuron_dimension.get_size())
    {}

    auto operator == (SelfType const& other) const
    {
        return som_dimension == other.som_dimension and
               neuron_dimension == other.neuron_dimension and
               data == other.data;
    }

    auto get_data_pointer() { return &data[0]; }
    auto get_data_pointer() const { return &data[0]; }

    auto get_neuron(SOMLayoutType const& position) {
    	auto&& beg = data.begin() + (position.dimension[0] * som_dimension.dimension[1] + position.dimension[1]) * neuron_dimension.get_size();
    	auto&& end = beg + neuron_dimension.get_size();
        return NeuronType(neuron_dimension, std::vector<T>(beg, end));
    }

    auto get_som_layout() -> SOMLayoutType { return som_dimension; }
    auto get_som_layout() const -> SOMLayoutType const { return som_dimension; }
    auto get_neuron_layout() -> NeuronLayoutType { return neuron_dimension; }
    auto get_neuron_layout() const -> NeuronLayoutType const { return neuron_dimension; }

    auto get_som_dimension() -> typename SOMLayoutType::DimensionType { return som_dimension.dimension; }
    auto get_som_dimension() const -> typename SOMLayoutType::DimensionType const { return som_dimension.dimension; }
    auto get_neuron_dimension() -> typename NeuronLayoutType::DimensionType { return neuron_dimension.dimension; }
    auto get_neuron_dimension() const -> typename NeuronLayoutType::DimensionType const { return neuron_dimension.dimension; }

private:

    template <typename A, typename B, typename C>
    friend void write(SOM<A, B, C> const& som, std::string const& filename);

    SOMLayoutType som_dimension;

    NeuronLayoutType neuron_dimension;

    std::vector<T> data;

    // Header of initialization SOM, will be copied to resulting SOM
    std::string header;

    thrust::device_vector<T> d_data;

};

template <>
SOM<CartesianLayout<1>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_dimension.get_size() * neuron_dimension.get_size())
{}

template <>
SOM<CartesianLayout<2>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width, input_data.som_height}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_dimension.get_size() * neuron_dimension.get_size())
{}

template <>
SOM<CartesianLayout<3>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width, input_data.som_height, input_data.som_depth}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_dimension.get_size() * neuron_dimension.get_size())
{}

template <>
SOM<HexagonalLayout, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_dimension.get_size() * neuron_dimension.get_size())
{}

} // namespace pink
