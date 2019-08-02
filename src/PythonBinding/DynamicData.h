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

namespace pink {

class DynamicData
{
public:

	DynamicData() {}

	DynamicData(DataType data_type, Layout layout, std::vector<size_t> dimensions, void* ptr)
	{
        if (data_type == DataType::FLOAT and layout == Layout::CARTESIAN and dimensions.size() == 2) {
        	auto p = static_cast<float*>(ptr);
        	std::vector<float> v(p, p + dimensions[0] * dimensions[1]);
        	data = std::make_shared<Data<CartesianLayout<2>, float>>(CartesianLayout<2>{dimensions[0], dimensions[1]}, v);
        }
	}

private:

	std::shared_ptr<DataBase> data;

};

} // namespace pink
