/**
 * @file   Point.cpp
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "Point.h"
#include <cmath>

bool isPositive(int n)
{
    return n >= 0;
}

std::ostream& operator << (std::ostream& os, Point p)
{
	return os << "(" << p.x << "," << p.y << ")";
}

float distance_square(Point pos1, Point pos2)
{
    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

float distance_hexagonal(Point pos1, Point pos2)
{
	int dx = pos1.x - pos2.x;
	int dy = pos1.y - pos2.y;

	if (isPositive(dx) == isPositive(dy))
	    return abs(dx + dy);
	else
	    return std::max(abs(dx), abs(dy));
}
