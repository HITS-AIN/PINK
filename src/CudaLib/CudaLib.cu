/**
 * @file   CudaLib/CudaLib.cu
 * @brief  Basic CUDA functions
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <sstream>

#include "CudaLib.h"

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
        for (int j = 0; j < 3; ++j)
            printf("  Maximum dimension %d of block:  %d\n", j, devProp.maxThreadsDim[j]);
        for (int j = 0; j < 3; ++j)
            printf("  Maximum dimension %d of grid:   %d\n", j, devProp.maxGridSize[j]);
        printf("  Clock rate:                    %d\n",  devProp.clockRate);
        printf("  Total constant memory:         %lu\n",  devProp.totalConstMem);
        printf("  Texture alignment:             %lu\n",  devProp.textureAlignment);
        printf("  Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("  Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("  Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
        printf("\n");
    }
}

std::vector<int> cuda_get_gpu_ids()
{
    std::vector<int> gpu_ids;

    int number_of_gpu_devices;
    cudaGetDeviceCount(&number_of_gpu_devices);

    char const* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");

    if (cuda_visible_devices == nullptr) {
        gpu_ids.resize(static_cast<size_t>(number_of_gpu_devices));
        std::iota(std::begin(gpu_ids), std::end(gpu_ids), 0);
    } else {
        // Split comma separated string into vector
        std::string token;
        for(std::stringstream ss(cuda_visible_devices); std::getline(ss, token, ',');)
            gpu_ids.push_back(std::stoi(token));
    }

    return gpu_ids;
}

} // namespace pink
