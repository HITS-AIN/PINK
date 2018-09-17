#!/usr/bin/python3

import numpy
import pink

som = pink.cartesian_2d_cartesian_2d_float();
image = pink.cartesian_2d_float(100,100);

print("SOM info:", som.info())
print("Image info:", image.info())
