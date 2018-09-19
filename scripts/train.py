#!/usr/bin/python3

import numpy as np
import pink

som = pink.cartesian_2d_cartesian_2d_float()
print("SOM info:", som.info())

a = np.array([[0.2, 0.4, 0.5],
              [0.7, 0.1, 0.3],
              [0.0, 1.0, 0.6]])

print(a.ndim)

image = pink.cartesian_2d_float()
print("Image info:", image.info())

trainer = pink.trainer()
trainer(som, a)
