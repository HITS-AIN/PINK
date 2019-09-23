/**
 * @file   UtilitiesLib/buffer_info.h
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

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
