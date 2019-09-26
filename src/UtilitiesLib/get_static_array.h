/**
 * @file   UtilitiesLib/get_static_array.h
 * @date   Sep 26, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <array>
#include <cassert>
#include <vector>

namespace pink {

/// Convert std::vector into std:array
template <size_t N, typename T>
std::array<T, N> get_static_array(std::vector<T> const& v)
{
    assert(v.size() >= N);
    std::array<T, N> a;
    for (size_t i = 0; i != N; ++i) a[i] = v[i];
    return a;
}

} // namespace pink
