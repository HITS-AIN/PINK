/**
 * @file   CudaLib/dot_dp4a.cu
 * @date   Apr 16, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#include <sm_61_intrinsics.h>

namespace pink {

typedef unsigned int uint;

__global__
void dot_dp4a_kernel(int *d_in1, int *d_in2, int *d_in3, int* d_out)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    int tx = threadIdx.x;

    d_out[tx] = __dp4a(d_in1[tx], d_in2[tx], d_in3[tx]);
#endif
}

void dot_dp4a(int *d_in1, int *d_in2, int *d_in3, int *d_out)
{
    dot_dp4a_kernel<<<1, 1>>>(d_in1, d_in2, d_in3, d_out);
    cudaDeviceSynchronize();
}

__global__
void dot_dp4a_kernel(uint *d_in1, uint *d_in2, uint *d_in3, uint* d_out)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    int tx = threadIdx.x;

    d_out[tx] = __dp4a(d_in1[tx], d_in2[tx], d_in3[tx]);
#endif
}

void dot_dp4a(uint *d_in1, uint *d_in2, uint *d_in3, uint *d_out)
{
    dot_dp4a_kernel<<<1, 1>>>(d_in1, d_in2, d_in3, d_out);
    cudaDeviceSynchronize();
}

} // namespace pink
