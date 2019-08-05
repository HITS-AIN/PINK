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

    DynamicData(DataType data_type, Layout layout, std::vector<ssize_t> dimensions, void* ptr)
     : data_type(data_type),
       layout(layout),
       dimensionality(dimensions.size())
    {
        std::vector<uint32_t> my_dimensions(std::begin(dimensions), std::end(dimensions));

        if (data_type == DataType::FLOAT and layout == Layout::CARTESIAN and dimensionality == 1)
        {
            auto p = static_cast<float*>(ptr);
            std::vector<float> v(p, p + dimensions[0]);
            data = std::make_shared<Data<CartesianLayout<1>, float>>(
                CartesianLayout<1>{my_dimensions[0]}, v);
        }
        else if (data_type == DataType::FLOAT and layout == Layout::CARTESIAN and dimensionality == 2)
        {
            auto p = static_cast<float*>(ptr);
            std::vector<float> v(p, p + dimensions[0] * dimensions[1]);

            data = std::make_shared<Data<CartesianLayout<2>, float>>(
                CartesianLayout<2>{my_dimensions[0], my_dimensions[1]}, v);
        }
        else if (data_type == DataType::FLOAT and layout == Layout::CARTESIAN and dimensionality == 3)
        {
            auto p = static_cast<float*>(ptr);
            std::vector<float> v(p, p + dimensions[0] * dimensions[1] * dimensions[2]);
            data = std::make_shared<Data<CartesianLayout<3>, float>>(
                CartesianLayout<3>{my_dimensions[0], my_dimensions[1], my_dimensions[2]}, v);
        }
        else
        {
            throw std::runtime_error("DynamicData: Unsupported type");
        }
    }

    buffer_info get_buffer_info() const
    {
        if (data_type == DataType::FLOAT and layout == Layout::CARTESIAN and dimensionality == 2)
        {
            auto&& dimension = std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data)->get_dimension();
            auto&& ptr = std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data)->get_data_pointer();

            return buffer_info(static_cast<void*>(ptr), static_cast<ssize_t>(sizeof(float)), "f", static_cast<ssize_t>(2),
                std::vector<ssize_t>{dimension[0], dimension[1]},
                std::vector<ssize_t>{sizeof(float) * dimension[1], sizeof(float)});
        }
        else
        {
            throw std::runtime_error("DynamicData: Unsupported type");
        }
    }

private:

    std::shared_ptr<DataBase> data;

    DataType data_type;
    Layout layout;
    uint8_t dimensionality;

};

} // namespace pink
