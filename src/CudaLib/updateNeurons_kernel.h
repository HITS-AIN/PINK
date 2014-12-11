/**
 * @file   CudaLib/updateNeurons_kernel.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "UtilitiesLib/InputData.h"

// Function pointers are also possible for the distribution- and distance function, but have a significant overhead.

__device__ bool isPositive(int n)
{
    return n >= 0;
}

struct DistributionFunctorBase
{
    __device__ virtual float operator ()(float distance) const = 0;
    virtual ~DistributionFunctorBase() {};
};

struct GaussianFunctor : public DistributionFunctorBase
{
    GaussianFunctor(float sigma) : sigma(sigma) {}

    ~GaussianFunctor() {};

    __device__ float operator ()(float distance) const
    {
        return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * pow((distance/sigma),2));
    }

private:

    float sigma;

};

struct MexicanHatFunctor : public DistributionFunctorBase
{
    MexicanHatFunctor(float sigma) : sigma(sigma), sigma2(sigma*sigma) {}

    ~MexicanHatFunctor() {};

    __device__ float operator ()(float distance) const
    {
        float distance2 = distance * distance;
        return 2.0 / (sqrt(3.0 * sigma) * pow(M_PI, 0.25)) * (1.0 - distance2/sigma2) * exp(-distance2 / (2.0 * sigma2));
    }

private:

    float sigma;
    float sigma2;

};

struct QuadraticDistanceFunctor
{
    __device__ float operator ()(int x1, int y1, int x2, int y2) const
    {
        return sqrt(powf(x1 - x2, 2) + powf(y1 - y2, 2));
    }
};

struct HexagonalDistanceFunctor
{
    __device__ float operator ()(int x1, int y1, int x2, int y2) const
    {
        int dx = x1 - x2;
        int dy = y1 - y2;

        if (isPositive(dx) == isPositive(dy))
            return abs(dx + dy);
        else
            return max(abs(dx), abs(dy));
    }
};

//! CUDA Kernel Device code updating quadratic self organizing map using gaussian function.
template <unsigned int block_size, class FunctionFunctor, class DistanceFunctor>
__global__ void
updateNeurons_kernel(float *som, float *rotatedImages, int *bestRotationMatrix, int *bestMatch,
    int neuron_size, FunctionFunctor functionFunctor, DistanceFunctor distanceFunctor, float damping, float maxUpdateDistance)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= neuron_size) return;

    int ij = blockIdx.z*gridDim.y + blockIdx.y;

    float distance = distanceFunctor(bestMatch[0], bestMatch[1], blockIdx.z, blockIdx.y);

    if (maxUpdateDistance <= 0.0 or distance < maxUpdateDistance)
    {
        float factor = functionFunctor(distance) * damping;
        som[ij*neuron_size + i] -= (som[ij*neuron_size + i] - rotatedImages[bestRotationMatrix[ij]*neuron_size + i]) * factor;
    }
}
