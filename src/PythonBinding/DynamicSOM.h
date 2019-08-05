/**
 * @file   PythonBinding/DynamicSOM.h
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>
#include <string>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/buffer_info.h"

namespace pink {

class DynamicSOM
{
public:

    DynamicSOM(std::string const& data_type, std::string const& som_layout,
        std::string const& neuron_layout, std::vector<ssize_t> shape, void* ptr);

    buffer_info get_buffer_info() const;

private:

    std::shared_ptr<SOMBase> data;

    std::string data_type;

    std::string som_layout;

    std::string neuron_layout;

    uint8_t dimensionality;

};

} // namespace pink
