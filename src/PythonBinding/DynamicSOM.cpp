/**
 * @file   PythonBinding/DynamicSOM.cpp
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cassert>

#include "DynamicSOM.h"

namespace pink {

DynamicSOM::DynamicSOM(std::string const& data_type, std::string const& som_layout,
        std::string const& neuron_layout, std::vector<uint32_t> const& shape, void* ptr)
 : m_data_type(data_type),
   m_som_layout(som_layout),
   m_neuron_layout(neuron_layout),
   m_shape(shape)
{
    if (m_data_type != "float32") throw std::runtime_error("data-type not supported");

    if (m_som_layout == "cartesian-2d")
    {
        assert(m_shape.size() >= 3);
        m_som = get_som<CartesianLayout<2>>({{m_shape[0], m_shape[1]}}, std::vector<uint32_t>(m_shape.begin() + 2, m_shape.end()),
        	static_cast<float*>(ptr), m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]);
    }
    else if (m_som_layout == "hexagonal-2d")
    {
        assert(m_shape.size() >= 2);
        auto dim = HexagonalLayout::get_dim_from_size(m_shape[0]);
        m_som = get_som<HexagonalLayout>({{dim, dim}}, std::vector<uint32_t>(m_shape.begin() + 1, m_shape.end()),
        	static_cast<float*>(ptr), m_shape[0] * m_shape[1] * m_shape[2]);
    }
    else
    {
        throw std::runtime_error("som_layout " + m_som_layout + " not supported");
    }
}

buffer_info DynamicSOM::get_buffer_info() const
{
    if (m_som_layout == "cartesian-2d")
    {
        auto&& som_shape = std::dynamic_pointer_cast<
            SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m_som)->get_som_dimension();
        auto&& neuron_shape = std::dynamic_pointer_cast<
            SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m_som)->get_neuron_dimension();
        auto&& ptr = std::dynamic_pointer_cast<
            SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m_som)->get_data_pointer();

        return buffer_info(static_cast<void*>(ptr), static_cast<ssize_t>(sizeof(float)),
            "f", static_cast<ssize_t>(4),
            std::vector<ssize_t>{static_cast<ssize_t>(som_shape[0]),
                                 static_cast<ssize_t>(som_shape[1]),
                                 static_cast<ssize_t>(neuron_shape[0]),
                                 static_cast<ssize_t>(neuron_shape[1])},
            std::vector<ssize_t>{static_cast<ssize_t>(sizeof(float) * neuron_shape[1] * neuron_shape[0] * som_shape[1]),
                                 static_cast<ssize_t>(sizeof(float) * neuron_shape[1] * neuron_shape[0]),
                                 static_cast<ssize_t>(sizeof(float) * neuron_shape[1]),
                                 static_cast<ssize_t>(sizeof(float))});
    }
    else if (m_som_layout == "hexagonal-2d")
    {
        auto&& number_of_neurons = std::dynamic_pointer_cast<
            SOM<HexagonalLayout, CartesianLayout<2>, float>>(m_som)->get_number_of_neurons();
        auto&& neuron_shape = std::dynamic_pointer_cast<
            SOM<HexagonalLayout, CartesianLayout<2>, float>>(m_som)->get_neuron_dimension();
        auto&& ptr = std::dynamic_pointer_cast<
            SOM<HexagonalLayout, CartesianLayout<2>, float>>(m_som)->get_data_pointer();

        return buffer_info(static_cast<void*>(ptr), static_cast<ssize_t>(sizeof(float)),
            "f", static_cast<ssize_t>(3),
            std::vector<ssize_t>{static_cast<ssize_t>(number_of_neurons),
                                 static_cast<ssize_t>(neuron_shape[0]),
                                 static_cast<ssize_t>(neuron_shape[1])},
            std::vector<ssize_t>{static_cast<ssize_t>(sizeof(float) * neuron_shape[1] * neuron_shape[0]),
                                 static_cast<ssize_t>(sizeof(float) * neuron_shape[1]),
                                 static_cast<ssize_t>(sizeof(float))});
    }
    else
    {
        throw std::runtime_error("som_layout " + m_som_layout + " not supported");
    }
}

} // namespace pink
