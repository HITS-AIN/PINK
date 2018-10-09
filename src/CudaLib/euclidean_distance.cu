/**
 * @file   CudaTest/mixed_precision.cpp
 * @date   Apr 16, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#include "sm_61_intrinsics.h"

namespace pink {

typedef unsigned int uint;

__global__
void cuda_euclidean_distance(int *d_in1, int *d_in2, int *d_in3, int* d_out)
{
	int tx = threadIdx.x;

#if __CUDA_ARCH__ >= 610
	d_out[tx] = __dp4a(d_in1[tx], d_in2[tx], d_in3[tx]);
#endif
}

void euclidean_distance_dp4a(int *d_in1, int *d_in2, int *d_in3, int *d_out, size_t size)
{
    cuda_euclidean_distance<<<1, 1>>>(d_in1, d_in2, d_in3, d_out);
    cudaDeviceSynchronize();
}

__global__
void cuda_euclidean_distance(uint *d_in1, uint *d_in2, uint *d_in3, uint* d_out)
{
	int tx = threadIdx.x;

#if __CUDA_ARCH__ >= 610
	d_out[tx] = __dp4a(d_in1[tx], d_in2[tx], d_in3[tx]);
#endif
}

void euclidean_distance_dp4a(uint *d_in1, uint *d_in2, uint *d_in3, uint *d_out, size_t size)
{
    cuda_euclidean_distance<<<1, 1>>>(d_in1, d_in2, d_in3, d_out);
    cudaDeviceSynchronize();
}

} // namespace pink
