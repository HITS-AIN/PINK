#!/usr/bin/python3

import numpy
import pink

som = pink.cartesian_2d_cartesian_2d_float()
print("SOM info:", som.info())

image = pink.cartesian_2d_float()
print("Image info:", image.info())

trainer = pink.trainer()
trainer(som, image)
