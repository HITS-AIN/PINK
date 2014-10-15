#!/usr/bin/python

import getopt
import Image
import matplotlib
import numpy
import struct
import sys
from matplotlib import pyplot

if __name__ == "__main__":

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    print 'Input file is ', inputfile
    print 'Output file is ', outputfile

    data = numpy.load(inputfile)

    print 'data.ndim = ', data.ndim
    print 'data.shape = ', data.shape
    print 'data.size = ', data.size

    of = open(outputfile, 'wb')

    for i in range(data.ndim):
        of.write(struct.pack('i', data.shape[i]))

    data.astype('f').tofile(of)
    of.close()

    print 'All done.'
    sys.exit()
