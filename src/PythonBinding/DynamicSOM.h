/**
 * @file   PythonBinding/DynamicSOM.h
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>
#include <string>

#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/buffer_info.h"

namespace pink {

struct DynamicSOM
{
    DynamicSOM(std::string const& data_type, std::string const& som_layout,
        std::string const& neuron_layout, std::vector<ssize_t> shape, void* ptr);

    buffer_info get_buffer_info() const;

    std::shared_ptr<SOMBase> m_data;

    std::string m_data_type;

    std::string m_som_layout;

    std::string m_neuron_layout;

    std::vector<ssize_t> m_shape;
};

} // namespace pink
