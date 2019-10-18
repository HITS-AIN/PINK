/**
 * @file   UtilitiesLib/buffer_info.h
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

struct buffer_info
{
    buffer_info(void *ptr, ssize_t itemsize, std::string format, ssize_t ndim,
        std::vector<ssize_t> shape, std::vector<ssize_t> strides)
    :
        m_ptr(ptr),
        m_itemsize(itemsize),
        m_format(format),
        m_ndim(ndim),
        m_shape(shape),
        m_strides(strides)
    {}

    void *m_ptr;
    ssize_t m_itemsize;
    std::string m_format;
    ssize_t m_ndim;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
};

} // namespace pink

inline std::ostream& operator << (std::ostream& os, pink::buffer_info const& buffer_info)
{
    os << "ptr = " << buffer_info.m_ptr << std::endl;
    os << "itemsize = " << buffer_info.m_itemsize << std::endl;
    os << "format = " << buffer_info.m_format << std::endl;
    os << "ndim = " << buffer_info.m_ndim << std::endl;
    os << "shape = ";
    for (size_t i = 0; i < buffer_info.m_shape.size(); ++i) os << buffer_info.m_shape[i] << " ";
    os << std::endl;
    os << "strides = ";
    for (size_t i = 0; i < buffer_info.m_strides.size(); ++i) os << buffer_info.m_strides[i] << " ";
    os << std::endl;
    return os;
}
