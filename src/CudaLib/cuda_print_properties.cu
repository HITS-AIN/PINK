/**
 * @file   CudaLib/cuda_print_properties.cu
 * @brief  Print device properties of GPU cards.
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include <stdio.h>

namespace pink {

void cuda_print_properties()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("  CUDA Device Query...\n");
    printf("  There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\n  CUDA Device #%d\n", i);

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        printf("  Major revision number:         %d\n",  devProp.major);
        printf("  Minor revision number:         %d\n",  devProp.minor);
        printf("  Name:                          %s\n",  devProp.name);
        printf("  Total global memory:           %lu\n",  devProp.totalGlobalMem);
        printf("  Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
        printf("  Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("  Warp size:                     %d\n",  devProp.warpSize);
        printf("  Maximum memory pitch:          %lu\n",  devProp.memPitch);
        printf("  Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
            printf("  Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("  Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("  Clock rate:                    %d\n",  devProp.clockRate);
        printf("  Total constant memory:         %lu\n",  devProp.totalConstMem);
        printf("  Texture alignment:             %lu\n",  devProp.textureAlignment);
        printf("  Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("  Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("  Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
        printf("\n");
    }
}

} // namespace pink
