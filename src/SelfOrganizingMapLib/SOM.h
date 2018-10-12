/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <stddef.h>
#include <array>
#include <functional>
#include <numeric>
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

    typedef T value_type;
    typedef SOM<SOMLayout, NeuronLayout, T> SelfType;
    typedef typename SOMLayout::DimensionType SOMDimensionType;
    typedef typename NeuronLayout::DimensionType NeuronDimensionType;
    typedef Data<NeuronLayout, T> NeuronType;

    /// Default construction
    SOM()
     : som_dimension{0},
       neuron_dimension{0}
    {}

    /// Construction by input data
    SOM(InputData const& input_data);

    /// Construction without initialization
    SOM(SOMDimensionType const& som_dimension, NeuronDimensionType const& neuron_dimension)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(get_size(som_dimension) * get_size(neuron_dimension))
    {}

    /// Construction and initialize all element to value
    SOM(SOMDimensionType const& som_dimension, NeuronDimensionType const& neuron_dimension, T value)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(get_size(som_dimension) * get_size(neuron_dimension), value)
    {}

    /// Construction and copy data into SOM
    SOM(SOMDimensionType const& som_dimension, NeuronDimensionType const& neuron_dimension, T* data)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(data, data + get_size(som_dimension) * get_size(neuron_dimension))
    {}

    bool operator == (SelfType const& other) const
    {
        return som_dimension == other.som_dimension and
               neuron_dimension == other.neuron_dimension and
               data == other.data;
    }

    T* get_data_pointer() { return &data[0]; }
    T const* get_data_pointer() const { return &data[0]; }

    NeuronType get_neuron(SOMDimensionType const& position) {
        return NeuronType(neuron_dimension, &data[(position[0] * som_dimension[1] + position[1]) * get_size(neuron_dimension)]);
    }

    SOMDimensionType get_som_dimension() const { return som_dimension; }
    NeuronDimensionType get_neuron_dimension() const { return neuron_dimension; }

private:

    template <typename T2, size_t dim>
    T2 get_size(std::array<T2, dim> const& dimension) {
        return std::accumulate(dimension.begin(), dimension.end(), 1, std::multiplies<T2>());
    }

    SOMDimensionType som_dimension;

    NeuronDimensionType neuron_dimension;

    std::vector<T> data;

};

template <>
SOM<CartesianLayout<1>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(get_size(som_dimension) * get_size(neuron_dimension))
{}

template <>
SOM<CartesianLayout<2>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width, input_data.som_height}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(get_size(som_dimension) * get_size(neuron_dimension))
{}

template <>
SOM<CartesianLayout<3>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width, input_data.som_height, input_data.som_depth}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(get_size(som_dimension) * get_size(neuron_dimension))
{}

template <>
SOM<HexagonalLayout, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_dimension{{input_data.som_width}},
   neuron_dimension{{input_data.neuron_dim, input_data.neuron_dim}},
   data(get_size(som_dimension) * get_size(neuron_dimension))
{}

} // namespace pink
