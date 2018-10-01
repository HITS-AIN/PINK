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

template <uint8_t dim>
struct CartesianLayout
{
    typedef typename std::array<uint32_t, dim> DimensionType;
};

struct HexagonalLayout
{
    typedef typename std::array<uint32_t, 1> DimensionType;
};


//! Primary template for generic SOM
template <typename SOMLayout, typename NeuronLayout, typename T>
class SOM_generic;

template <typename T>
class SOM_generic<CartesianLayout<2>, CartesianLayout<2>, T>
{
public:

    typedef T value_type;
    typedef SOM_generic<CartesianLayout<2>, CartesianLayout<2>, T> SelfType;
    typedef CartesianLayout<2> SOMLayout;
    typedef CartesianLayout<2> NeuronLayout;
    typedef Cartesian<2, T> NeuronType;

    /// Default construction
    SOM_generic()
     : som_dimension{0},
       neuron_dimension{0}
    {}

    /// Construction without initialization
    SOM_generic(SOMLayout::DimensionType const& som_dimension, NeuronLayout::DimensionType const& neuron_dimension)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(std::accumulate(som_dimension.begin(), som_dimension.end(), 1, std::multiplies<uint32_t>()) *
            std::accumulate(neuron_dimension.begin(), neuron_dimension.end(), 1, std::multiplies<uint32_t>()))
    {}

    /// Construction and initialize all element to value
    SOM_generic(SOMLayout::DimensionType const& som_dimension, NeuronLayout::DimensionType const& neuron_dimension, T value = 0.0)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(std::accumulate(som_dimension.begin(), som_dimension.end(), 1, std::multiplies<uint32_t>()) *
            std::accumulate(neuron_dimension.begin(), neuron_dimension.end(), 1, std::multiplies<uint32_t>()), value)
    {}

    /// Construction and copy data into SOM
    SOM_generic(SOMLayout::DimensionType const& som_dimension, NeuronLayout::DimensionType const& neuron_dimension, T* data)
     : som_dimension(som_dimension),
       neuron_dimension(neuron_dimension),
       data(data, data +
            std::accumulate(som_dimension.begin(), som_dimension.end(), 1, std::multiplies<uint32_t>()) *
            std::accumulate(neuron_dimension.begin(), neuron_dimension.end(), 1, std::multiplies<uint32_t>()))
    {}

    bool operator == (SelfType const& other) const
    {
        return som_dimension == other.som_dimension and
               neuron_dimension == other.neuron_dimension and
               data == other.data;
    }

    T* get_data_pointer() { return &data[0]; }
    T const* get_data_pointer() const { return &data[0]; }

    NeuronType get_neuron(SOMLayout::DimensionType const& position) {
        return NeuronType(neuron_dimension, &data[(position[0] * som_dimension[1] + position[1]) * get_size<2>(neuron_dimension)]);
    }

    SOMLayout::DimensionType get_som_dimension() const { return som_dimension; }
    NeuronLayout::DimensionType get_neuron_dimension() const { return neuron_dimension; }

private:

    template <uint8_t dim>
    uint32_t get_size(std::array<uint32_t, dim> const& dimension) const {
        return std::accumulate(dimension.begin(), dimension.end(), 1, std::multiplies<uint32_t>());
    }

    SOMLayout::DimensionType som_dimension;

    NeuronLayout::DimensionType neuron_dimension;

    std::vector<T> data;

};

} // namespace pink
