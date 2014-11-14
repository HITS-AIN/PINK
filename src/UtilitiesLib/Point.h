/**
 * @file   Point.h
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef POINT_H_
#define POINT_H_

#include <iostream>

struct Point
{
	Point(int x = 0, int y = 0) : x(x), y(y) {}

	int x;
	int y;
};

//! Pretty printing of Point.
std::ostream& operator << (std::ostream& os, Point point);

//! Return the distance of two points.
float distance(Point pos1, Point pos2);

#endif /* POINT_H_ */
