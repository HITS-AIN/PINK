#!/usr/bin/python3
"""
PINK Training of SOM
"""

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import pink

__author__  = "Bernd Doser"
__email__   = "bernd.doser@h-its.org"
__license__ = "GPLv3"

def gaussian(distance, sigma=1.0):
    """ Returns the value of an gaussian distribution """
    return 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * math.pow((distance / sigma), 2))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PINK SOM training')
    parser.add_argument('inputfile')
    parser.add_argument('-d', '--display', action='store_true', help='Display SOM during training')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    
    args = parser.parse_args()
    print('Input file:', args.inputfile)
    print('Display:', args.display)
    print('Verbose:', args.verbose)

    images = np.load(args.inputfile).astype(np.float32)
    if args.verbose:
        print('Image shape: ', images.shape, ', dtype: ', images.dtype)
        print('Image shape[0]: ', images.shape[0])

    np_som = np.ndarray(shape=(3, 3, 64, 64), dtype=np.float32)
    som = pink.cartesian_2d_cartesian_2d_float(np_som)

    for i in range(images.shape[0]):

        if args.display:
            plt.matshow(images[i])
            plt.show()
            
        image = pink.cartesian_2d_float(images[i])

        trainer = pink.trainer()
        trainer(som, image)
    
    trained_som = np.array(som, copy=True)
