/**
 * @file   Point.h
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

namespace pink {

struct Point
{
    Point(int x = 0, int y = 0) : x(x), y(y) {}

    int x;
    int y;
};

//! Pretty printing of Point.
std::ostream& operator << (std::ostream& os, Point p);

} // namespace pink
