#!/usr/bin/python

import getopt
import Image
import matplotlib
import numpy
import struct
import sys
from matplotlib import pyplot

if __name__ == "__main__":

    data = numpy.array([[[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]], [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]], numpy.float)

    print 'data.ndim = ', data.ndim
    print 'data.shape = ', data.shape
    print 'data.size = ', data.size

    of = open('test.bin', 'wb')

    for i in range(data.ndim):
        of.write(struct.pack('i', data.shape[i]))

    data.astype('f').tofile(of)
    of.close()

    print 'All done.'
    sys.exit()
