/**
 * @file   pink/DynamicSOM.cpp
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
    if (m_data_type != "float32") throw pink::exception("data-type not supported");

    if (m_som_layout == "cartesian-2d")
    {
        assert(m_shape.size() >= 3);
        m_som = get_som<CartesianLayout<2>>({{m_shape[0], m_shape[1]}}, std::vector<uint32_t>(m_shape.begin() + 2, m_shape.end()),
            static_cast<float*>(ptr), std::accumulate(m_shape.begin(), m_shape.end(), 1U, std::multiplies<uint32_t>()));
    }
    else if (m_som_layout == "hexagonal-2d")
    {
        assert(m_shape.size() >= 2);
        auto dim = HexagonalLayout::get_dim_from_size(m_shape[0]);
        m_som = get_som<HexagonalLayout>({{dim, dim}}, std::vector<uint32_t>(m_shape.begin() + 1, m_shape.end()),
            static_cast<float*>(ptr), std::accumulate(m_shape.begin(), m_shape.end(), 1U, std::multiplies<uint32_t>()));
    }
    else
    {
        throw pink::exception("SOM layout " + m_som_layout + " is not supported");
    }
}

buffer_info DynamicSOM::get_buffer_info() const
{
    if (m_som_layout == "cartesian-2d")
    {
        return get_buffer_info<CartesianLayout<2>>();
    }
    else if (m_som_layout == "hexagonal-2d")
    {
        return get_buffer_info<HexagonalLayout>();
    }
    else
    {
        throw pink::exception("SOM layout " + m_som_layout + " is not supported");
    }
}

} // namespace pink
