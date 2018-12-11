/**
 * @file   CudaLib/CudaLib.h
 * @brief  Basic CUDA functions
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <vector>

namespace pink {

/// Print CUDA device properties
void cuda_print_properties();

/// Return IDs of available GPU devices in CUDA_VISIBLE_DEVICES
std::vector<int> cuda_get_gpu_ids();

} // namespace pink
