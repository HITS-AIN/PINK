/**
 * @file   Point.cpp
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "Point.h"
#include <cmath>

std::ostream& operator << (std::ostream& os, Point p)
{
	return os << "(" << p.x << "," << p.y << ")";
}

float distance(Point pos1, Point pos2)
{
    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}
