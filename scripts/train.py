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

class GaussianFunctor(object):
    """ Returns the value of an gaussian distribution """
    
    def __init__(self, sigma = 1.1, damping = 0.2): 
        self.sigma = sigma
        self.damping = damping 

    def __call__(self, distance):
        return self.damping / (self.sigma * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * math.pow((distance / self.sigma), 2))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PINK SOM training')
    parser.add_argument('images', nargs='+', help='Input file of images')
    parser.add_argument('-i', '--input-som', help='Input file of SOM initialization')
    parser.add_argument('-o', '--output-som', help='Output file of resulting SOM')
    parser.add_argument('--som-dim', type=int, default=5, help='Dimension of SOM if initialized from scratch')
    parser.add_argument('-d', '--display', action='store_true', help='Display SOM during training')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    
    args = parser.parse_args()
    if args.verbose:
        print('Images file:', args.images)
        print('Input SOM file:', args.input_som)
        print('Output SOM file:', args.output_som)
        print('SOM dimension:', args.som_dim)
        print('Display:', args.display)

    images = np.load(args.images[0]).astype(np.float32)
    for input in args.images[1:]:
        images = np.append(images, np.load(input).astype(np.float32), axis = 0)

    image_dim = images.shape[1]
    neuron_dim = int(image_dim * math.sqrt(2.0) / 2.0)

    # Randomize order of input images
    np.random.shuffle(images)

    if args.verbose:
        print('Image shape: ', images.shape, ', dtype: ', images.dtype)

    if args.input_som:
        # Initialize SOM by input file
        np_som = np.load(args.input_som).astype(np.float32)
    else:
        # Initialize SOM by random values
        np_som = np.random.rand(args.som_dim, args.som_dim, neuron_dim, neuron_dim).astype(np.float32)

    if args.verbose:
        print('SOM shape: ', np_som.shape, ', dtype: ', np_som.dtype)

    if args.display:
        new_dim = np_som.shape[0] * np_som.shape[2]
        plt.matshow(np_som.swapaxes(1,2).reshape((new_dim, new_dim)))
        plt.show()

    som = pink.som_cartesian_2d_cartesian_2d_float(np_som)
    trainer = pink.trainer(distribution_function = GaussianFunctor(sigma = 1.1, damping = 1.0),
                           number_of_rotations = 180, verbosity = 0, use_gpu = False)

    for i in range(images.shape[0]):

        image = pink.cartesian_2d_float(images[i])
        trainer(som, image)

        np_som = np.array(som, copy = False)

        if args.display and i % 100 == 0:
            new_dim = np_som.shape[0] * np_som.shape[2]
            plt.matshow(np_som.swapaxes(1,2).reshape((new_dim, new_dim)))
            plt.show()

    if args.output_som:
        np.save(args.output_som, np_som)

    print('All done.')
