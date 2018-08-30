/**
 * @file   CudaTest/mixed_precision.cpp
 * @date   Apr 16, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#include "sm_61_intrinsics.h"

namespace pink {

__global__
void cuda_euclidean_distance(float *a1, float *a2, size_t size)
{
#if __CUDA_ARCH__ >= 610

    //unsigned int __dp4a(a1, a2, c);

#else

#endif
}

void euclidean_distance(float *a1, float *a2, size_t size)
{
    int8_t *ia;
    cudaMalloc((void **) &ia, size * sizeof(int8_t));

    //cuda_convert_float_to_int8<<<1, 1>>>(ia, a1, size);

    cuda_euclidean_distance<<<1, 1>>>(a1, a2, size);
}

} // namespace pink
