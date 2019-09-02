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

    np_data = np.array([[1, 2], [3, 4]], np.float32)
    print(np_data)

    data = pink.data(np_data, data_type="float32", layout="cartesian-2d")

    np_som = np.zeros((2, 2, 2, 2)).astype(np.float32)
    print(np_som)

    som = pink.som(np_som, data_type="float32", som_layout="cartesian-2d", neuron_layout="cartesian-2d")
    
    trainer = pink.trainer(som, GaussianFunctor(sigma=1.1, damping=1.0), number_of_rotations=1, use_flip=False, use_gpu=True)
    trainer(data)
    trainer.update_som()
    
    np_som_res = np.array(som, copy=False)
    print(np_som_res)

    print('All done.')

main()
