/**
 * @file   UtilitiesLib/DistanceFunctors.h
 * @brief  Virtual functors for distance functions used by updating SOM.
 * @date   Dec 4, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef DISTANCEFUNCTOR_H_
#define DISTANCEFUNCTOR_H_

#include <algorithm>
#include <cmath>

/**
 * @brief Abstract base for distance functor.
 */
struct DistanceFunctorBase
{
    virtual float operator () (int x1, int y1, int x2, int y2) const = 0;
    virtual ~DistanceFunctorBase() {}
};

/**
 * @brief Functor for distance in quadratic layout.
 */
struct QuadraticDistanceFunctor : public DistanceFunctorBase
{
    float operator () (int x1, int y1, int x2, int y2) const
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    }
};

/**
 * @brief Functor for distance in hexagonal layout.
 */
struct HexagonalDistanceFunctor : public DistanceFunctorBase
{
    float operator () (int x1, int y1, int x2, int y2) const
    {
        int dx = x1 - x2;
        int dy = y1 - y2;

        if (isPositive(dx) == isPositive(dy))
            return std::abs(dx + dy);
        else
            return std::max(std::abs(dx), std::abs(dy));
    }

private:

    bool isPositive(int n) const { return n >= 0; }

};

#endif /* DISTANCEFUNCTOR_H_ */
