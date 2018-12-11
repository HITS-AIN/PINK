/**
 * @file   rotateAndCropTexture.cu
 * @date   Nov 13, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

texture<float, 2, cudaReadModeElementType> image_texture;

/**
 * CUDA Kernel Device code for combined rotation and cropping of a list of images.
 */
template <unsigned int block_size>
__global__ void
rotateAndCropTexture_kernel(float *rotatedImages, float *image, int neuron_size,
    int neuron_dim, int image_dim, float *cosAlpha, float *sinAlpha)
{
    int x2 = blockIdx.x * blockDim.x + threadIdx.x;
    int y2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (x2 >= neuron_dim or y2 >= neuron_dim) return;

    int x0 = image_dim * 0.5;
    int y0 = image_dim * 0.5;
    int margin = (image_dim - neuron_dim) * 0.5;
    int x0margin = x0 - margin;
    int y0margin = y0 - margin;

    float cosAlpha_local = cosAlpha[blockIdx.z];
    float sinAlpha_local = sinAlpha[blockIdx.z];

    int x1 = (x2-x0margin)*cosAlpha_local + (y2-y0margin)*sinAlpha_local + x0;
    int y1 = (y2-y0margin)*cosAlpha_local - (x2-x0margin)*sinAlpha_local + y0;

    float *pCurRot = rotatedImages + blockIdx.z * neuron_size;

    //pCurRot[x*neuron_dim + y] = tex2D(image_texture, tx+0.5f, ty+0.5f);

    if (x1 >= 0 and x1 < image_dim and y1 >= 0 and y1 < image_dim) {
        atomicAdd(pCurRot + x2*neuron_dim + y2, image[x1*image_dim + y1]);
    } else {
        atomicAdd(pCurRot + x2*neuron_dim + y2, 0.0f);
    }
}

} // namespace pink
