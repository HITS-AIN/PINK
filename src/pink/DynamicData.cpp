/**
 * @file   pink/DynamicData.cpp
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cassert>

#include "DynamicData.h"
#include "UtilitiesLib/pink_exception.h"
#include "UtilitiesLib/get_strides.h"

namespace pink {

DynamicData::DynamicData(std::string const& data_type, std::string const& layout, std::vector<uint32_t> shape, void* ptr)
 : m_data_type(data_type),
   m_layout(layout),
   m_shape(shape)
{
    if (m_data_type != "float32") throw pink::exception("data-type not supported");
    auto p = static_cast<float*>(ptr);
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    if (m_layout == "cartesian-1d") {
        assert(m_shape.size() == 1);
        m_data = std::make_shared<Data<CartesianLayout<1U>, float>>(
            CartesianLayout<1U>{m_shape[0]},
            std::vector<float>(p, p + size));
    } else if (m_layout == "cartesian-2d") {
        assert(m_shape.size() == 2);
        m_data = std::make_shared<Data<CartesianLayout<2U>, float>>(
            CartesianLayout<2U>{m_shape[0], m_shape[1]},
            std::vector<float>(p, p + size));
    } else if (m_layout == "cartesian-3d") {
        assert(m_shape.size() == 3);
        m_data = std::make_shared<Data<CartesianLayout<3U>, float>>(
            CartesianLayout<3U>{m_shape[0], m_shape[1], m_shape[2]},
            std::vector<float>(p, p + size));
    } else {
        throw pink::exception("layout " + m_layout + " is not supported");
    }
}

buffer_info DynamicData::get_buffer_info() const
{
    void* ptr = nullptr;

    if (m_layout == "cartesian-1d") {
        ptr = static_cast<void*>(std::dynamic_pointer_cast<
            Data<CartesianLayout<1U>, float>>(m_data)->get_data_pointer());
    } else if (m_layout == "cartesian-2d") {
        ptr = static_cast<void*>(std::dynamic_pointer_cast<
            Data<CartesianLayout<2U>, float>>(m_data)->get_data_pointer());
    } else if (m_layout == "cartesian-3d") {
        ptr = static_cast<void*>(std::dynamic_pointer_cast<
            Data<CartesianLayout<3U>, float>>(m_data)->get_data_pointer());
    } else {
        throw pink::exception("layout " + m_layout + " is not supported");
    }

    return buffer_info(ptr, static_cast<ssize_t>(sizeof(float)),
        "f", static_cast<ssize_t>(m_shape.size()),
        std::vector<ssize_t>(m_shape.begin(), m_shape.end()), get_strides(m_shape));
}

} // namespace pink
