/**
 * @file   UtilitiesLib/DistanceFunctors.h
 * @brief  Virtual functors for distance functions used by updating SOM.
 * @date   Dec 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef DISTANCEFUNCTOR_H_
#define DISTANCEFUNCTOR_H_

#include "Error.h"
#include <algorithm>
#include <cmath>

/**
 * @brief Abstract base for distance functor.
 *
 * Calculate the distance between two points p1 and p2.
 * Position is given as index of a continuous array.
 */
struct DistanceFunctorBase
{
    virtual float operator () (int p1, int p2, int dim) const = 0;
    virtual ~DistanceFunctorBase() {}
};

/**
 * @brief Functor for distance in quadratic layout.
 */
struct QuadraticDistanceFunctor : public DistanceFunctorBase
{
    float operator () (int p1, int p2, int dim) const
    {
        int x1 = p1 / dim;
        int y1 = p1 % dim;
        int x2 = p2 / dim;
        int y2 = p2 % dim;
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    }
};

/**
 * @brief Functor for distance in hexagonal layout.
 */
struct HexagonalDistanceFunctor : public DistanceFunctorBase
{
    float operator () (int p1, int p2, int dim) const
    {
        int x1, y1, x2, y2;
        getHexagonalIndices(p1, dim, x1, y1);
        getHexagonalIndices(p2, dim, x2, y2);

        int dx = x1 - x2;
        int dy = y1 - y2;

        if (isPositive(dx) == isPositive(dy))
            return std::abs(dx + dy);
        else
            return std::max(std::abs(dx), std::abs(dy));
    }

private:

    bool isPositive(int n) const { return n >= 0; }

    void getHexagonalIndices(int p, int dim, int &x, int &y) const
    {
        int radius = (dim - 1)/2;
        int pos = 0;
        for (x = -radius; x <= radius; ++x) {
            for (y = -radius - std::min(0,x); y <= radius - std::max(0,x); ++y, ++pos) {
                if (pos == p) return;
            }
        }
        fatalError("Error in hexagonal indices.");
    }

};

#endif /* DISTANCEFUNCTOR_H_ */
