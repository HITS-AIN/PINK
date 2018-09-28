/**
 * @file   UtilitiesLib/DistanceFunctors.h
 * @brief  Virtual functors for distance functions used by updating SOM.
 * @date   Dec 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "Error.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace pink {

/**
 * @brief Abstract base for distance functor.
 */
struct DistanceFunctorBase
{
#ifdef __CUDACC__
	__device__
#endif
    virtual float operator () (int p1, int p2) const = 0;

    virtual ~DistanceFunctorBase() {}
};

//! Primary template should never be instantiated.
template <int dim, bool periodic = false>
struct CartesianDistanceFunctor;

/**
 * @brief Calculate the distance in a non-periodic one-dimensional cartesian grid.
 */
template <>
struct CartesianDistanceFunctor<1, false> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width)
     : width_(width)
    {}

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
    {
        return abs(p1 - p2);
    }

private:

    int width_;

};

/**
 * @brief Calculate the distance in a periodic one-dimensional cartesian grid.
 */
template <>
struct CartesianDistanceFunctor<1, true> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width)
     : width_(width)
    {}

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
    {
        int dx = abs(p1 - p2);
        //return std::min(abs(delta), std::min(abs(delta + width_), abs(delta - width_)));
        return dx > width_ * 0.5 ? width_ - dx : dx;
    }

private:

    int width_;

};

/**
 * @brief Calculate the distance in a non-periodic two-dimensional cartesian grid.
 */
template <>
struct CartesianDistanceFunctor<2, false> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height)
     : width_(width), height_(height)
    {}

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
    {
        int y1 = p1 / height_;
        int x1 = p1 % height_;
        int y2 = p2 / height_;
        int x2 = p2 % height_;
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    }

private:

    int width_;
    int height_;

};

/**
 * @brief Calculate the distance in a periodic two-dimensional cartesian grid.
 */
template <>
struct CartesianDistanceFunctor<2, true> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height)
     : width_(width), height_(height)
    {}

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
    {
        int y1 = p1 / height_;
        int x1 = p1 % height_;
        int y2 = p2 / height_;
        int x2 = p2 % height_;
        int dx = abs(x1 - x2);
        int dy = abs(y1 - y2);
        if (dx > width_ * 0.5) dx = width_ - dx;
        if (dy > height_ * 0.5) dy = height_ - dy;
        return sqrt(pow(dx, 2) + pow(dy, 2));
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

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
    {
        int offset = height_ * width_;
        int z1 = p1 / offset;
        int y1 = (p1 - z1 * offset) / height_;
        int x1 = p1 % height_;
        int z2 = p2 / offset;
        int y2 = (p2 - z2 * offset) / height_;
        int x2 = p2 % height_;
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
    }

private:

    int width_;
    int height_;
    int depth_;

};

/**
 * @brief Calculate the distance in a periodic three-dimensional cartesian grid.
 */
template <>
struct CartesianDistanceFunctor<3, true> : public DistanceFunctorBase
{
    CartesianDistanceFunctor(int width, int height, int depth)
     : width_(width), height_(height), depth_(depth)
    {}

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
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
        return sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
    }

private:

    int width_;
    int height_;
    int depth_;

};

/**
 * @brief Calculate the distance in hexagonal grid.
 */
struct HexagonalDistanceFunctor : public DistanceFunctorBase
{
    HexagonalDistanceFunctor(int dim)
        : dim_(dim)
       {}

#ifdef __CUDACC__
	__device__
#endif
    float operator () (int p1, int p2) const
    {
        int x1, y1, x2, y2;
        getHexagonalIndices(p1, x1, y1);
        getHexagonalIndices(p2, x2, y2);

        int dx = x1 - x2;
        int dy = y1 - y2;

        if (isPositive(dx) == isPositive(dy))
#ifdef __CUDACC__
            return abs(dx + dy);
#else
            return std::abs(dx + dy);
#endif
        else
#ifdef __CUDACC__
            return max(abs(dx), abs(dy));
#else
            return std::max(std::abs(dx), std::abs(dy));
#endif
    }

private:

#ifdef __CUDACC__
	__device__
#endif
    bool isPositive(int n) const { return n >= 0; }

#ifdef __CUDACC__
	__device__
#endif
    void getHexagonalIndices(int p, int &x, int &y) const
    {
        int radius = (dim_ - 1)/2;
        int pos = 0;
        for (x = -radius; x <= radius; ++x) {
#ifdef __CUDACC__
            for (y = -radius - min(0,x); y <= radius - max(0,x); ++y, ++pos) {
#else
            for (y = -radius - std::min(0,x); y <= radius - std::max(0,x); ++y, ++pos) {
#endif
                if (pos == p) return;
            }
        }
#ifndef __CUDACC__
        fatalError("Error in hexagonal indices.");
#endif
    }

    int dim_;

};

} // namespace pink
