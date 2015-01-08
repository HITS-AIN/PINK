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
    __device__ virtual float operator () (float distance) const = 0;
    virtual ~DistributionFunctorBase() {};
};

struct GaussianFunctor : public DistributionFunctorBase
{
    GaussianFunctor(float sigma) : sigma(sigma) {}

    ~GaussianFunctor() {};

    __device__ float operator () (float distance) const
    {
        return 1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * powf((distance/sigma),2));
    }

private:

    float sigma;

};

struct MexicanHatFunctor : public DistributionFunctorBase
{
    MexicanHatFunctor(float sigma) : sigma(sigma), sigma2(sigma*sigma) {}

    ~MexicanHatFunctor() {};

    __device__ float operator () (float distance) const
    {
        float distance2 = distance * distance;
        return 2.0 / (sqrt(3.0 * sigma) * powf(M_PI, 0.25)) * (1.0 - distance2/sigma2) * exp(-distance2 / (2.0 * sigma2));
    }

private:

    float sigma;
    float sigma2;

};

__device__ void getHexagonalIndices(int p, int dim, int &x, int &y)
{
    int radius = (dim - 1)/2;
    int pos = 0;
    for (x = -radius; x <= radius; ++x) {
        for (y = -radius - min(0,x); y <= radius - max(0,x); ++y, ++pos) {
            if (pos == p) return;
        }
    }
}

//! Abstract base for distance functor.
struct DistanceFunctorBase
{
    __device__ virtual float operator () (int p1, int p2) const = 0;
    virtual ~DistanceFunctorBase() {}
};

//! Primary template should never be instantiated.
template <int dim, bool periodic = false>
struct CartesianDistanceFunctor;

//! Calculate the distance in a non-periodic one-dimensional cartesian grid.
template <>
struct CartesianDistanceFunctor<1, false> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width)
     : width_(width)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        return abs(p1 - p2);
    }

private:

    int width_;

};

//! Calculate the distance in a periodic one-dimensional cartesian grid.
template <>
struct CartesianDistanceFunctor<1, true> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width)
     : width_(width)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        int dx = abs(p1 - p2);
        //return std::min(abs(delta), std::min(abs(delta + width_), abs(delta - width_)));
        return dx > width_ * 0.5 ? width_ - dx : dx;
    }

private:

    int width_;

};

//! Calculate the distance in a non-periodic two-dimensional cartesian grid.
template <>
struct CartesianDistanceFunctor<2, false> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height)
     : width_(width), height_(height)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        int y1 = p1 / height_;
        int x1 = p1 % height_;
        int y2 = p2 / height_;
        int x2 = p2 % height_;
        return sqrt(powf(x1 - x2, 2) + powf(y1 - y2, 2));
    }

private:

    int width_;
    int height_;

};

//! Calculate the distance in a periodic two-dimensional cartesian grid.
template <>
struct CartesianDistanceFunctor<2, true> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height)
     : width_(width), height_(height)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        int y1 = p1 / height_;
        int x1 = p1 % height_;
        int y2 = p2 / height_;
        int x2 = p2 % height_;
        int dx = abs(x1 - x2);
        int dy = abs(y1 - y2);
        if (dx > width_ * 0.5) dx = width_ - dx;
        if (dy > height_ * 0.5) dy = height_ - dy;
        return sqrt(powf(dx, 2) + powf(dy, 2));
    }

private:

    int width_;
    int height_;

};

/**
 * @brief Calculate the distance in a non-periodic three-dimensional cartesian grid.
 *
 * Calculate the distance between two points p1(x,y,z) and p2(x,y,z).
 * Position is given as index of a continuous array (z * height * width + y * height + x).
 */
template <>
struct CartesianDistanceFunctor<3, false> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height, int depth)
     : width_(width), height_(height), depth_(depth)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        int offset = height_ * width_;
        int z1 = p1 / offset;
        int y1 = (p1 - z1 * offset) / height_;
        int x1 = p1 % height_;
        int z2 = p2 / offset;
        int y2 = (p2 - z2 * offset) / height_;
        int x2 = p2 % height_;
        return sqrt(powf(x1 - x2, 2) + powf(y1 - y2, 2) + powf(z1 - z2, 2));
    }

private:

    int width_;
    int height_;
    int depth_;

};

//! Calculate the distance in a periodic three-dimensional cartesian grid.
template <>
struct CartesianDistanceFunctor<3, true> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height, int depth)
     : width_(width), height_(height), depth_(depth)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        int offset = height_ * width_;
        int z1 = p1 / offset;
        int y1 = (p1 - z1 * offset) / height_;
        int x1 = p1 % height_;
        int z2 = p2 / offset;
        int y2 = (p2 - z2 * offset) / height_;
        int x2 = p2 % height_;
        int dx = abs(x1 - x2);
        int dy = abs(y1 - y2);
        int dz = abs(z1 - z2);
        if (dx > width_ * 0.5) dx = width_ - dx;
        if (dy > height_ * 0.5) dy = height_ - dy;
        if (dz > height_ * 0.5) dz = height_ - dz;
        return sqrt(powf(dx, 2) + powf(dy, 2) + powf(dz, 2));
    }

private:

    int width_;
    int height_;
    int depth_;

};

//! Calculate the distance in hexagonal grid.
struct HexagonalDistanceFunctor : public DistanceFunctorBase
{
    HexagonalDistanceFunctor(int dim)
     : dim_(dim)
    {}

    __device__ float operator () (int p1, int p2) const
    {
        int x1, y1, x2, y2;
        getHexagonalIndices(p1, dim_, x1, y1);
        getHexagonalIndices(p2, dim_, x2, y2);

        int dx = x1 - x2;
        int dy = y1 - y2;

        if (isPositive(dx) == isPositive(dy))
            return abs(dx + dy);
        else
            return max(abs(dx), abs(dy));
    }
private:

    int dim_;

};

//! CUDA Kernel Device code updating quadratic self organizing map using gaussian function.
template <unsigned int block_size, class FunctionFunctor, class DistanceFunctor>
__global__ void
updateNeurons_kernel(float *som, float *rotatedImages, int *bestRotationMatrix, int *bestMatch,
    int neuron_size, FunctionFunctor functionFunctor, DistanceFunctor distanceFunctor,
    float damping, float maxUpdateDistance)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= neuron_size) return;

    float distance = distanceFunctor(*bestMatch, blockIdx.y);
    int pos = blockIdx.y * neuron_size + i;

    if (maxUpdateDistance <= 0.0 or distance < maxUpdateDistance)
    {
        float factor = functionFunctor(distance) * damping;
        som[pos] -= (som[pos] - rotatedImages[bestRotationMatrix[blockIdx.y] * neuron_size + i]) * factor;
    }
}
