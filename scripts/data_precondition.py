#!/usr/bin/env python3

"""
PINK data analysis and preconditioning
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import argparse
import numpy as np
import os
import tools

def main():
    """ Main routine of PINK data preconditioning """

    parser = argparse.ArgumentParser(description='PINK data preconditioning')
    parser.add_argument('data', help='Data input file (.npy or .bin)', action=tools.check_extension({'npy','bin'}))
    parser.add_argument('-o', '--output', help='Data output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    parser.add_argument('-s', '--scale', action='store_true', help='Scale the input data to be within the range [0, 1]')
    args = parser.parse_args()

    if os.path.splitext(args.data)[1][1:] == "npy":
        data = np.load(args.data).astype(np.float32)
    elif os.path.splitext(args.data)[1][1:] == "bin":
        data = tools.load_data(args.data)

    print('shape:             ', np.shape(data))
    print('size:              ', data.size)
    print('min value:         ', np.amin(data))
    print('max value:         ', np.amax(data))
    print('non-zero elements: ', np.count_nonzero(data))
    print('sparsity:          ', np.count_nonzero(data) / data.size)

    if args.scale:

        print('Data will be linearly scaled to be within the range [0.0, 1.0]') 

        min_element = np.amin(data)
        max_element = np.amax(data)
        factor = 1 / (max_element - min_element)
        
        print('min value: ', min_element)
        print('max value: ', max_element)
        print('factor: ', factor)
        
        data = (data - min_element) * factor
    
        print('min value: ', np.amin(data))
        print('max value: ', np.amax(data))

    if args.output:
        print('Output file written at', args.output) 
        if os.path.splitext(args.output)[1][1:] == "npy":
            np.save(args.output, data)
        elif os.path.splitext(args.output)[1][1:] == "bin":
            tools.save_data(args.output, data)
        else:
            raise RuntimeError('Unsupported output file extension: ', os.path.splitext(args.output)[1][1:])

    print('All done.')

main()
