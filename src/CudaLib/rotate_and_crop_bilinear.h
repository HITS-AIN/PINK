/**
 * @file   CudaLib/rotate_and_crop_bilinear.h
 * @date   Nov 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code for combined rotation and cropping of a list of quadratic images.
 */
template <unsigned int block_size>
__global__ void
rotate_and_crop_bilinear(float *rotatedImages, float *image, int neuron_size,
    int neuron_dim, int image_dim, float *cosAlpha, float *sinAlpha, int numberOfChannels)
{
    int x2 = blockIdx.x * blockDim.x + threadIdx.x;
    int y2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (x2 >= neuron_dim or y2 >= neuron_dim) return;

    float center = (image_dim - 1) * 0.5;
    float margin = (image_dim - neuron_dim) * 0.5;
    float center_margin = center - margin;

    float cosAlpha_local = cosAlpha[blockIdx.z];
    float sinAlpha_local = sinAlpha[blockIdx.z];

    float x1 = (x2-center_margin)*cosAlpha_local + (y2-center_margin)*sinAlpha_local + center + 0.1;
    float y1 = (y2-center_margin)*cosAlpha_local - (x2-center_margin)*sinAlpha_local + center + 0.1;

    int ix1 = x1;
    int iy1 = y1;
    int ix1b = ix1 + 1;
    int iy1b = iy1 + 1;

    float rx1 = x1 - ix1;
    float ry1 = y1 - iy1;
    float cx1 = 1.0f - rx1;
    float cy1 = 1.0f - ry1;

    float *pCurRot = rotatedImages + blockIdx.z * numberOfChannels * neuron_size;

    float value = cx1 * cy1 * image[ix1  * image_dim + iy1 ]
                + cx1 * ry1 * image[ix1  * image_dim + iy1b]
                + rx1 * cy1 * image[ix1b * image_dim + iy1 ]
                + rx1 * ry1 * image[ix1b * image_dim + iy1b];

    if (x1 >= 0 and x1 < image_dim and y1 >= 0 and y1 < image_dim) {
        atomicExch(pCurRot + x2*neuron_dim + y2, value);
    } else {
        atomicExch(pCurRot + x2*neuron_dim + y2, 0.0f);
    }
}
