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
    parser.add_argument('images', help='Input file of images')
    parser.add_argument('som', help='Output file of SOM')
    parser.add_argument('-d', '--display', action='store_true', help='Display SOM during training')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    
    args = parser.parse_args()
    print('Images file:', args.images)
    print('SOM file:', args.som)
    print('Display:', args.display)
    print('Verbose:', args.verbose)

    images = np.load(args.images).astype(np.float32)
    if args.verbose:
        print('Image shape: ', images.shape, ', dtype: ', images.dtype)

#     np_som = np.array([[images[0], images[1], images[2]],
#                        [images[3], images[4], images[5]],
#                        [images[6], images[7], images[8]]], dtype = np.float32)
    np_som = np.random.rand(3, 3, 44, 44).astype(np.float32)
    if args.verbose:
        print('SOM shape: ', np_som.shape, ', dtype: ', np_som.dtype)
        
    if args.display:
        new_dim = np_som.shape[0] * np_som.shape[2]
        plt.matshow(np_som.swapaxes(1,2).reshape((new_dim, new_dim)))
        plt.show()

    som = pink.cartesian_2d_cartesian_2d_float(np_som)
    trainer = pink.trainer(number_of_rotations = 90, verbosity = 1)

    for i in range(images.shape[0]):

        if args.display:
            plt.matshow(images[i])
            plt.show()

        image = pink.cartesian_2d_float(images[i])

        trainer(som, image)

        np_som = np.array(som, copy=True)
        if args.verbose:
            print('SOM shape: ', np_som.shape, ', dtype: ', np_som.dtype)

        if args.display:
            new_dim = np_som.shape[0] * np_som.shape[2]
            plt.matshow(np_som.swapaxes(1,2).reshape((new_dim, new_dim)))
            plt.show()

    print('\n  Successfully finished. Have a nice day.\n')
