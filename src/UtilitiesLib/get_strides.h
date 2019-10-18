/**
 * @file   UtilitiesLib/get_strides.h
 * @date   Oct 2, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <vector>

namespace pink {

inline std::vector<ssize_t> get_strides(std::vector<uint32_t> shape)
{
    std::vector<ssize_t> strides(shape.size(), sizeof(float));
    for (size_t i = 1; i < shape.size(); ++i)
        for (size_t j = i; j < shape.size(); ++j) strides[i - 1] *= shape[j];
    return strides;
}

} // namespace pink
