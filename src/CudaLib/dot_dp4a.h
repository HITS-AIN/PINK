/**
 * @file   CudaLib/dot_dp4a.h
 * @date   Apr 17, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#pragma once

#include <cstddef>

namespace pink {

typedef unsigned int uint;

void dot_dp4a(int *d_in1, int *d_in2, int *d_in3, int *d_out);

void dot_dp4a(uint *d_in1, uint *d_in2, uint *d_in3, uint *d_out);

} // namespace pink
