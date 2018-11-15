/**
 * @file   CudaLib/CudaLib.h
 * @brief  Header for all CUDA functions
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <thrust/device_vector.h>

namespace pink {

//! Print CUDA device properties.
void cuda_print_properties();

//! CUDA test routine for image rotation.
void cuda_rotate(int height, int width, float *source, float *dest, float angle);

//! Basic allocation for device memory.
float* cuda_alloc_float(int size);

//! Basic allocation for device memory.
int* cuda_alloc_int(int size);

//! Basic allocation for device memory.
uint* cuda_alloc_uint(int size);

//! Fill device memory with zero.
void cuda_fill_zero(float *d, int size);

//! Free device memory.
void cuda_free(float *d);

//! Free device memory.
void cuda_free(int *d);

//! Copy memory from host to device.
void cuda_copyHostToDevice_float(float *dest, float *source, int size);

//! Copy memory from host to device.
void cuda_copyHostToDevice_int(int *dest, int *source, int size);

//! Copy memory from host to device.
void cuda_copyHostToDevice_uint(uint *dest, uint *source, int size);

//! Copy memory from device to host.
void cuda_copyDeviceToHost_float(float *dest, float *source, int size);

//! Copy memory from device to host.
void cuda_copyDeviceToHost_int(int *dest, int *source, int size);

//! Copy memory from device to host.
void cuda_copyDeviceToHost_uint(uint *dest, uint *source, int size);

//! Return number of GPUs.
int cuda_getNumberOfGPUs();

//! Set GPU device number.
void cuda_setDevice(int number);

} // namespace pink
