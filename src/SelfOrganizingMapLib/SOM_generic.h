/**
 * @file   SelfOrganizingMapLib/SOM_generic.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Cartesian.h"

namespace pink {

template <size_t dim>
struct CartesianLayout
{
	static const size_t dimensionality = dim;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;
};

struct HexagonalLayout
{
	static const size_t dimensionality = 1;
    typedef typename std::array<uint32_t, dimensionality> DimensionType;
};


//! Primary template for generic SOM
template <typename SOMLayout, typename NeuronLayout, typename T>
class SOM_generic
{
public:

    typedef T value_type;
    typedef SOM_generic<SOMLayout, NeuronLayout, T> SelfType;
    typedef typename SOMLayout::DimensionType SOMDimensionType;
    typedef typename NeuronLayout::DimensionType NeuronDimensionType;
    typedef Cartesian<NeuronLayout::dimensionality, T> NeuronType;

    /// Default construction
    SOM_generic()
     : som_dimension{0},
       neuron_dimension{0}
    {}

    /// Construction without initialization
    SOM_generic(SOMDimensionType const& som_dimension, NeuronDimensionType const& neuron_dimension)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(get_size(som_dimension) * get_size(neuron_dimension))
    {}

    /// Construction and initialize all element to value
    SOM_generic(SOMDimensionType const& som_dimension, NeuronDimensionType const& neuron_dimension, T value = 0.0)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(get_size(som_dimension) * get_size(neuron_dimension), value)
    {}

    /// Construction and copy data into SOM
    SOM_generic(SOMDimensionType const& som_dimension, NeuronDimensionType const& neuron_dimension, T* data)
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

} // namespace pink
