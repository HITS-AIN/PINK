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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

} // namespace pink
