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
    parser.add_argument('som', help='Output file of SOM')
    parser.add_argument('images', nargs='+', help='Input file of images')
    parser.add_argument('-d', '--display', action='store_true', help='Display SOM during training')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    
    args = parser.parse_args()
    print('Images file:', args.images)
    print('SOM file:', args.som)
    print('Display:', args.display)
    print('Verbose:', args.verbose)

    images = np.load(args.images[0]).astype(np.float32)
    for input in args.images[1:]:
        print(input)
        images = np.append(images, np.load(input).astype(np.float32), axis = 0)

    np.random.shuffle(images)

    if args.verbose:
        print('Image shape: ', images.shape, ', dtype: ', images.dtype)

    np_som = np.random.rand(5, 5, 44, 44).astype(np.float32)
    if args.verbose:
        print('SOM shape: ', np_som.shape, ', dtype: ', np_som.dtype)
        
    if args.display:
        new_dim = np_som.shape[0] * np_som.shape[2]
        plt.matshow(np_som.swapaxes(1,2).reshape((new_dim, new_dim)))
        plt.show()

    som = pink.som_cartesian_2d_cartesian_2d_float(np_som)
    trainer = pink.trainer(number_of_rotations = 180, verbosity = args.verbose)

    #for i in range(5):
    for i in range(images.shape[0]):

        image = pink.cartesian_2d_float(images[i])
        trainer(som, image)

        np_som = np.array(som, copy = False)

        if args.display and i % 100 == 0:
            new_dim = np_som.shape[0] * np_som.shape[2]
            plt.matshow(np_som.swapaxes(1,2).reshape((new_dim, new_dim)))
            plt.show()

    print('All done.')
