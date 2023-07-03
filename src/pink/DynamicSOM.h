/**
 * @file   pink/DynamicSOM.h
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/get_strides.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/buffer_info.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

struct DynamicSOM
{
    DynamicSOM(std::string const& data_type, std::string const& som_layout,
        std::string const& neuron_layout, std::vector<uint32_t> const& shape, void* ptr);

    buffer_info get_buffer_info() const;

    template <typename SOM_Layout>
    auto get_som(SOM_Layout const& som_layout, std::vector<uint32_t> const& neuron_shape,
        float* p, uint32_t size) -> std::shared_ptr<SOMBase>
    {
        if (m_neuron_layout == "cartesian-1d") {
            assert(neuron_shape.size() == 1);
            return std::make_shared<SOM<SOM_Layout, CartesianLayout<1U>, float>>(
                som_layout,
                CartesianLayout<1U>{{neuron_shape[0]}},
                std::vector<float>(p, p + size));
        } else if (m_neuron_layout == "cartesian-2d") {
            assert(neuron_shape.size() == 2);
            return std::make_shared<SOM<SOM_Layout, CartesianLayout<2U>, float>>(
                som_layout,
                CartesianLayout<2U>{{neuron_shape[0], neuron_shape[1]}},
                std::vector<float>(p, p + size));
        } else if (m_neuron_layout == "cartesian-3d") {
            assert(neuron_shape.size() == 3);
            return std::make_shared<SOM<SOM_Layout, CartesianLayout<3U>, float>>(
                som_layout,
                CartesianLayout<3U>{{neuron_shape[0], neuron_shape[1], neuron_shape[2]}},
                std::vector<float>(p, p + size));
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }
    }

    template <typename SOM_Layout>
    buffer_info get_buffer_info() const
    {
        void* ptr = nullptr;

        if (m_neuron_layout == "cartesian-1d") {
            ptr = static_cast<void*>(std::dynamic_pointer_cast<
                SOM<SOM_Layout, CartesianLayout<1U>, float>>(m_som)->get_data_pointer());
        } else if (m_neuron_layout == "cartesian-2d") {
            ptr = static_cast<void*>(std::dynamic_pointer_cast<
                SOM<SOM_Layout, CartesianLayout<2U>, float>>(m_som)->get_data_pointer());
        } else if (m_neuron_layout == "cartesian-3d") {
            ptr = static_cast<void*>(std::dynamic_pointer_cast<
                SOM<SOM_Layout, CartesianLayout<3U>, float>>(m_som)->get_data_pointer());
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }

        return buffer_info(ptr, static_cast<ssize_t>(sizeof(float)),
            "f", static_cast<ssize_t>(m_shape.size()),
            std::vector<ssize_t>(m_shape.begin(), m_shape.end()), get_strides(m_shape));
    }

    std::shared_ptr<SOMBase> m_som;

    std::string m_data_type;

    std::string m_som_layout;

    std::string m_neuron_layout;

    std::vector<uint32_t> m_shape;
};

} // namespace pink
