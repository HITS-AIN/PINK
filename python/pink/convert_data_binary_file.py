#!/usr/bin/env python3

"""
PINK convert data binary file from version 1 to version 2
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import argparse
import os
import struct
import tools

def main():
    """ Main routine of PINK convert data binary file """

    parser = argparse.ArgumentParser(description='PINK convert binary formats')
    parser.add_argument('data', help='Data input file (.bin)', action=tools.check_extension({'bin'}))
    parser.add_argument('-o', '--output', default="out.bin", help='Data output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    args = parser.parse_args()

    input = open(args.data, 'rb')
    header = tools.get_header_comments(input)
    
    nb_images, nb_channels, width, height = struct.unpack('i' * 4, input.read(4 * 4))
    size = nb_channels * width * height
    
    # <file format version> 0 <data-type> <number of entries> <data layout> <data>
    output = open(args.output, 'wb')
    output.write(header)

    if nb_channels == 1:
        output.write(struct.pack('i' * 8, 2, 0, 0, nb_images, 0, 2, width, height))
    else:
        output.write(struct.pack('i' * 9, 2, 0, 0, nb_images, 0, 3, width, height, nb_channels))

    for _ in range(nb_images):
        output.write(input.read(size*4))

    print('All done.')

main()
