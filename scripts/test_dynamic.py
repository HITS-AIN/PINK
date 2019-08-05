#!/usr/bin/env python3

"""
PINK Training of SOM
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import math
import numpy as np
import pink
import tools

class GaussianFunctor():
    """ Returns the value of a Gaussian distribution """

    def __init__(self, sigma=1.1, damping=0.2):
        self.sigma = sigma
        self.damping = damping

    def __call__(self, distance):
        return self.damping / (self.sigma * math.sqrt(2.0 * math.pi)) \
            * math.exp(-0.5 * math.pow((distance / self.sigma), 2))


def main():

    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)
    print(image)

    data_cart = pink.data(pink.data_type.FLOAT, pink.layout.CARTESIAN, image)

    np_data_cart = np.array(data_cart, copy=False)
    print(np_data_cart)

    print('All done.')

main()
