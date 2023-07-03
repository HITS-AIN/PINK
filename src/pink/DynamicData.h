/**
 * @file   pink/DynamicData.h
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

struct DynamicData
{
    DynamicData(std::string const& data_type, std::string const& layout, std::vector<uint32_t> shape, void* ptr);

    buffer_info get_buffer_info() const;

    std::shared_ptr<DataBase> m_data;

    std::string m_data_type;

    std::string m_layout;

    std::vector<uint32_t> m_shape;
};

} // namespace pink
