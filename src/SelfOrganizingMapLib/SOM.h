/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <functional>
#include <vector>

#ifdef __CUDACC__
    #include <thrust/device_vector.h>
#endif

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
     : som_layout{0},
       neuron_layout{0}
    {}

    /// Construction by input data
    SOM(InputData const& input_data);

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

    auto get_data_pointer() { return &data[0]; }
    auto get_data_pointer() const { return &data[0]; }

#ifdef __CUDACC__
    /// Return device vector
    auto get_device_vector() -> thrust::device_vector<T>& { return d_data; }
    auto get_device_vector() const -> thrust::device_vector<T> const& { return d_data; }

    /// Return device pointer
    auto get_device_pointer() { return &d_data[0]; }
    auto get_device_pointer() const { return &d_data[0]; }

    void update_host() { data = d_data; }
    void update_device() { d_data = data; }
#endif

    auto get_neuron(SOMLayoutType const& position) {
        auto&& beg = data.begin() + (position.dimension[0] * som_layout.dimension[1] + position.dimension[1]) * neuron_layout.size();
        auto&& end = beg + neuron_layout.size();
        return NeuronType(neuron_layout, std::vector<T>(beg, end));
    }

    auto get_number_of_neurons() -> uint32_t const { return som_layout.size(); }
    auto get_neuron_size() -> uint32_t const { return neuron_layout.size(); }

    auto get_som_layout() -> SOMLayoutType { return som_layout; }
    auto get_som_layout() const -> SOMLayoutType const { return som_layout; }
    auto get_neuron_layout() -> NeuronLayoutType { return neuron_layout; }
    auto get_neuron_layout() const -> NeuronLayoutType const { return neuron_layout; }

    auto get_som_dimension() -> typename SOMLayoutType::DimensionType { return som_layout.dimension; }
    auto get_som_dimension() const -> typename SOMLayoutType::DimensionType const { return som_layout.dimension; }
    auto get_neuron_dimension() -> typename NeuronLayoutType::DimensionType { return neuron_layout.dimension; }
    auto get_neuron_dimension() const -> typename NeuronLayoutType::DimensionType const { return neuron_layout.dimension; }

private:

    template <typename A, typename B, typename C>
    friend void write(SOM<A, B, C> const& som, std::string const& filename);

    template <typename A, typename B, typename C>
    friend std::ostream& operator << (std::ostream& os, SOM<A, B, C> const& som);

    SOMLayoutType som_layout;

    NeuronLayoutType neuron_layout;

    // Header of initialization SOM, will be copied to resulting SOM
    std::string header;

#ifdef __CUDACC__
    thrust::device_vector<T> d_data;
    thrust::host_vector<T> data;
#else
    std::vector<T> data;
#endif

};

} // namespace pink
