#!/usr/bin/env python3

"""
PINK Training of SOM
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pink
from tqdm import tqdm

class GaussianFunctor():
    """ Returns the value of a Gaussian distribution """

    def __init__(self, sigma=1.1, damping=0.2):
        self.sigma = sigma
        self.damping = damping

    def __call__(self, distance):
        return self.damping / (self.sigma * math.sqrt(2.0 * math.pi)) \
            * math.exp(-0.5 * math.pow((distance / self.sigma), 2))


def main():
    """ Main routine of PINK training """

    print('PINK version ', pink.__version__)

    parser = argparse.ArgumentParser(description='PINK SOM training')
    parser.add_argument('images', nargs='+', help='Input file of images')
    parser.add_argument('-i', '--input-som', help='Input file of SOM initialization')
    parser.add_argument('-o', '--output-som', help='Output file of resulting SOM')
    parser.add_argument('--som-dim', type=int, default=8, help='Dimension of SOM if initialized from scratch')
    parser.add_argument('--neuron-dim', type=int, default=-1, help='Dimension of neurons')
    parser.add_argument('-d', '--display', action='store_true', help='Display SOM during training')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    parser.add_argument('-s', '--scale', action='store_true', help='Scale the input images to be within the range [0, 1]')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs [default: 1]')

    args = parser.parse_args()
    if args.verbose:
        print('Images file:', args.images)
        print('Input SOM file:', args.input_som)
        print('Output SOM file:', args.output_som)
        print('SOM dimension:', args.som_dim)
        print('Display:', args.display)

    images = np.load(args.images[0]).astype(np.float32)
    for image_file in args.images[1:]:
        images = np.append(images, np.load(image_file).astype(np.float32), axis=0)

    # Remove channels
    if len(images.shape) == 4 and images.shape[1] == 1:
        images = np.squeeze(images, axis=1)

    image_dim = images.shape[1]
    neuron_dim = args.neuron_dim if args.neuron_dim != -1 else int(image_dim / math.sqrt(2.0) * 2.0)
    euclid_dim = int(image_dim * math.sqrt(2.0) / 2.0)
    print('Neuron dimension:', neuron_dim)
    print('Euclid dimension:', euclid_dim)

    if args.scale:
        min_element = np.amin(images)
        max_element = np.amax(images)
        factor = 1 / (max_element - min_element)

        print('min value: ', min_element)
        print('max value: ', max_element)
        print('factor: ', factor)

        images = (images - min_element) * factor

    print('min value: ', np.amin(images))
    print('max value: ', np.amax(images))

    # Randomize order of input images
    np.random.shuffle(images)

    if args.verbose:
        print('Image shape: ', images.shape, ', dtype: ', images.dtype)

    if args.input_som:
        # Initialize SOM by input file
        np_som = np.load(args.input_som).astype(np.float32)
    else:
        # Initialize SOM by random values
        np_som = np.random.rand(
            args.som_dim, args.som_dim, neuron_dim, neuron_dim).astype(np.float32)

    if args.verbose:
        print('SOM shape: ', np_som.shape, ', dtype: ', np_som.dtype)

    if args.display:
        new_dim = np_som.shape[0] * np_som.shape[2]
        plt.matshow(np_som.swapaxes(1, 2).reshape((new_dim, new_dim)))
        plt.show()

    som = pink.SOM(np_som)

    trainer = pink.Trainer(som, euclidean_distance_dim=euclid_dim, verbosity=0,
                           distribution_function=pink.GaussianFunctor(sigma=1.1, damping=1.0),
                           number_of_rotations=360, interpolation=pink.Interpolation.BILINEAR,
                           euclidean_distance_type=pink.DataType.UINT8)

    for _ in tqdm(range(args.epochs), desc="epoch"):
        for i in tqdm(range(images.shape[0]), desc="train", leave=False):

            data = pink.Data(images[i])
            trainer(data)

            if args.display and i % 100 == 0:
                trainer.update_som()
                np_som = np.array(som, copy=False)
                new_dim = np_som.shape[0] * np_som.shape[2]
                plt.matshow(np_som.swapaxes(1, 2).reshape((new_dim, new_dim)))
                plt.show()

    if args.output_som:
        trainer.update_som()
        np_som = np.array(som, copy=False)
        np.save(args.output_som, np_som)

    print('All done.')

main()
