#!/usr/bin/env python3

"""
PINK test image rotation
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import math
import matplotlib.pyplot as plt
import numpy as np
import pink

def main():
    """ Main routine of PINK test image rotation """

    print('PINK version ', pink.__version__)

    images = np.load("../data/all_shapes_shuffled/scaled_images.npy").astype(np.float32)
    
    data = pink.data(images[0, 0, 0:, 0:])

    plt.matshow(images[0, 0, 0:, 0:])
    plt.show()

    print('All done.')

main()
