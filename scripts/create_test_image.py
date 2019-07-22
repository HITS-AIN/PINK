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
    parser.add_argument('--number-of-images', type=int, default=1, help='Number of images, default = 1')
    parser.add_argument('--image-dim', type=int, default=64, help='Dimension of images (quadratic, default = 64)')
    parser.add_argument('--fill-with', choices=['random', 'zero', 'one'], default='random', help='Pixel value (random=default, zero, one)')
    parser.add_argument('--file-version', choices=['1', '2'], default='2', help='File version (1, 2 = default)')
    args = parser.parse_args()

    # <file format version> 0 <data-type> <number of entries> <data layout> <data>
    output = open(args.output, 'wb')

    if args.file_version == '1':
        output.write(struct.pack('i' * 4, args.number_of_images, 1, args.image_dim, args.image_dim))
    elif args.file_version == '2':
        output.write(struct.pack('i' * 8, 2, 0, 0, args.number_of_images, 0, 2, args.image_dim, args.image_dim))
    
    if args.fill_with == 'random':
        floatlist = [random.uniform(0.0, 1.0) for _ in range(args.number_of_images * args.image_dim * args.image_dim)]
        output.write(struct.pack('%sf' % len(floatlist), *floatlist))
    elif args.fill_with == 'one':
        floatlist = [1.0 for _ in range(args.number_of_images * args.image_dim * args.image_dim)]
        output.write(struct.pack('%sf' % len(floatlist), *floatlist))

    print('All done.')

main()
