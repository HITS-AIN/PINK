#!/usr/bin/python

import getopt
import numpy
import matplotlib.pylab as plt
import struct
import sys

if __name__ == "__main__":

    inputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:",["ifile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    print 'Input file is ', inputfile

    inFile = open(inputfile, 'rb')
    size = struct.unpack('iii', inFile.read(12))

    print 'size = ', size

    array = numpy.array(struct.unpack('f'*size[1]*size[2], inFile.read(size[1]*size[2]*4)))
    data = numpy.ndarray([size[1],size[2]], 'float', array)

    inFile.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    print 'All done.'
    sys.exit()
