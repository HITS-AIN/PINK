/**
 * @file   PythonBinding/DynamicSOM.h
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/buffer_info.h"

namespace pink {

struct DynamicSOM
{
    DynamicSOM(std::string const& data_type, std::string const& som_layout,
        std::string const& neuron_layout, std::vector<uint32_t> const& shape, void* ptr);

    buffer_info get_buffer_info() const;

    template <typename SOM_Layout>
    auto get_som(SOM_Layout const& som_layout, std::vector<uint32_t> const& shape, float* p, uint32_t size)
        -> std::shared_ptr<SOMBase>
    {
        if (m_neuron_layout == "cartesian-1d") {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        } else if (m_neuron_layout == "cartesian-2d") {
            return std::make_shared<SOM<SOM_Layout, CartesianLayout<2>, float>>(
            	som_layout,
                CartesianLayout<2>{{m_shape[0], m_shape[1]}},
                std::vector<float>(p, p + size));
        } else if (m_neuron_layout == "cartesian-3d") {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }
    }

    std::shared_ptr<SOMBase> m_som;

    std::string m_data_type;

    std::string m_som_layout;

    std::string m_neuron_layout;

    std::vector<uint32_t> m_shape;
};

} // namespace pink
