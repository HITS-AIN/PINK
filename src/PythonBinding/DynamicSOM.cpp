/**
 * @file   PythonBinding/DynamicSOM.cpp
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include "DynamicSOM.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"

namespace pink {

DynamicSOM::DynamicSOM(std::string const& data_type, std::string const& som_layout,
        std::string const& neuron_layout, std::vector<ssize_t> shape, void* ptr)
 : m_data_type(data_type),
   m_som_layout(som_layout),
   m_neuron_layout(neuron_layout),
   m_shape(shape)
{
    if (data_type != "float32") throw std::runtime_error("data-type not supported");
    if (som_layout != "cartesian-2d") throw std::runtime_error("som_layout not supported");
    if (neuron_layout != "cartesian-2d") throw std::runtime_error("neuron_layout not supported");

    std::vector<uint32_t> my_shape(std::begin(shape), std::end(shape));

    if (m_shape.size() == 4)
    {
        auto&& p = static_cast<float*>(ptr);
        m_data = std::make_shared<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(
            CartesianLayout<2>{my_shape[0], my_shape[1]},
            CartesianLayout<2>{my_shape[2], my_shape[3]},
            std::vector<float>(p, p + my_shape[0] * my_shape[1] * my_shape[2] * my_shape[3]));
    }
    else
    {
        throw std::runtime_error("shape not supported");
    }
}

buffer_info DynamicSOM::get_buffer_info() const
{
    if (m_shape.size() == 4)
    {
        auto&& som_shape = std::dynamic_pointer_cast<
            SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m_data)->get_som_dimension();
        auto&& neuron_shape = std::dynamic_pointer_cast<
            SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m_data)->get_neuron_dimension();
        auto&& ptr = std::dynamic_pointer_cast<
            SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m_data)->get_data_pointer();

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
    else
    {
        throw std::runtime_error("shape not supported");
    }
}

} // namespace pink
