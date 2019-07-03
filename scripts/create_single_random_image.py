#!/usr/bin/env python3

"""
PINK create binary test data set filled with random numbers
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import argparse
import random
import os
import struct
import tools

def main():

    parser = argparse.ArgumentParser(description='PINK create binary test data set filled with random numbers')
    parser.add_argument('-o', '--output', default='data.bin', help='Data output file', action=tools.check_extension({'bin'}))
    parser.add_argument('--number-of-images', type=int, default=1, help='Number of images')
    parser.add_argument('--image-dim', type=int, default=64, help='Dimension of images (quadratic)')
    args = parser.parse_args()

    # <file format version> 0 <data-type> <number of entries> <data layout> <data>
    output = open(args.output, 'wb')
    output.write(struct.pack('i' * 8, 2, 0, 0, args.number_of_images, 0, 2, args.image_dim, args.image_dim))
    
    floatlist = [random.uniform(0.0, 1.0) for _ in range(args.number_of_images * args.image_dim * args.image_dim)]
    output.write(struct.pack('%sf' % len(floatlist), *floatlist))

    print('All done.')

main()
