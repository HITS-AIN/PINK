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
        ptr(ptr),
        itemsize(itemsize),
        format(format),
        ndim(ndim),
        shape(shape),
        strides(strides)
    {}

    void *ptr;
    ssize_t itemsize;
    std::string format;
    ssize_t ndim;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
};

} // namespace pink
