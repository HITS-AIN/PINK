#!/usr/bin/python3

import numpy as np
import math
import pink

def gaussian(distance, sigma = 1.0):
    """ Returns the value of an gaussian distribution """
    return 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * math.pow((distance/sigma), 2))

print(gaussian(1.0))

np_image = np.array([[0.2, 0.4, 0.5],
                     [0.7, 0.1, 0.3],
                     [0.0, 1.0, 0.6]], dtype = np.float32)

np_som = np.array([[np_image, np_image, np_image],
                   [np_image, np_image, np_image],
                   [np_image, np_image, np_image]], dtype = np.float32)

print(np_image.ndim)
print(np_som.ndim)

som = pink.cartesian_2d_cartesian_2d_float()
print("SOM info:", som.info())

image = pink.cartesian_2d_float()
print("Image info:", image.info())

trainer = pink.trainer()
trainer(som, image)

print(np_image[0,0])
print(np_image.dtype)
m = pink.Matrix(np_image)
np_m = np.array(m, copy = True)

print(np_m)

#trainer(som, np_image)
