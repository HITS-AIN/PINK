/**
 * @file   PythonBinding/DynamicData.h
 * @date   Aug 2, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/Data.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/buffer_info.h"

namespace pink {

class DynamicData
{
public:

    DynamicData(std::string const& data_type, std::string const& layout, std::vector<ssize_t> shape, void* ptr);

    buffer_info get_buffer_info() const;

private:

    std::shared_ptr<DataBase> data;

    std::string data_type;

    std::string layout;

    uint8_t dimensionality;

};

} // namespace pink
