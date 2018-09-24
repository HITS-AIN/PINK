#!/usr/bin/python3
"""
PINK Training of SOM
"""

import getopt
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import pink

__author__  = "Bernd Doser"
__email__   = "bernd.doser@h-its.org"
__license__ = "GPLv3"

def print_usage():
    print('')
    print('Usage:')
    print('')
    print('  train.py [Options] <inputfile>')
    print('')
    print('Options:')
    print('')
    print('  --display, -d          Display SOM during training.')
    print('  --help, -h             Print this lines.')
    print('  --verbose, -v          Be talkative.')
    print('')

def gaussian(distance, sigma=1.0):
    """ Returns the value of an gaussian distribution """
    return 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * math.pow((distance / sigma), 2))

if __name__ == "__main__":
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:h:v:", ["display", "help", "verbose"])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    display = False
    verbose = False

    # Use inputted parameters
    for opt, arg in opts:
        if opt in ("-d", "--display"):
            display = True
        elif opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = True
        else:
            print_usage()
            print ('ERROR: Unhandled option')

    if len(args) != 1:
        print_usage()
        print ('ERROR: Input file is missing.')
        sys.exit(1)

    inputfile = args[0]

    print('Input file:', inputfile)
    print('Display:', display)
    print('Verbose:', verbose)

    images = np.load(inputfile).astype(np.float32)
    if verbose:
        print('Image shape: ', images.shape, ', dtype: ', images.dtype)
        print('Image shape[0]: ', images.shape[0])

    np_som = np.ndarray(shape=(3, 3, 64, 64), dtype=np.float32)
    som = pink.cartesian_2d_cartesian_2d_float(np_som)

    for i in range(images.shape[0]):

        if display:
            plt.matshow(images[i])
            plt.show()
            
        image = pink.cartesian_2d_float(images[i])

        trainer = pink.trainer()
        trainer(som, image)
    
    trained_som = np.array(som, copy=True)
