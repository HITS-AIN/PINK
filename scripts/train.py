#!/usr/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt
import pink

def gaussian(distance, sigma = 1.0):
    """ Returns the value of an gaussian distribution """
    return 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * math.pow((distance/sigma), 2))

print(gaussian(1.0))

np_image = np.array([[0.2, 0.4, 0.5],
                     [0.7, 0.1, 0.3],
                     [0.0, 1.0, 0.6]], dtype = np.float32)

plt.matshow(np_image)
plt.show()

np_som = np.array([[np_image, np_image, np_image],
                   [np_image, np_image, np_image],
                   [np_image, np_image, np_image]], dtype = np.float32)

som = pink.cartesian_2d_cartesian_2d_float(np_som)
image = pink.cartesian_2d_float(np_image)

trainer = pink.trainer()
trainer(som, image)

trained_som = np.array(som, copy = True)
