#!/usr/bin/python

import Image
import matplotlib
import numpy
import struct
import sys
from matplotlib import pyplot

files = sys.argv[1:]
of = open('result.bin', 'wb')

data = numpy.load(files[0])
total = data.shape[0]
height = data.shape[1]
width = data.shape[2]

for file in files[1:]:
    data = numpy.load(file)
    total += data.shape[0]
    if (height != data.shape[1]):
        print 'Shape error.'
        sys.exit(1)
    if (width != data.shape[2]):
        print 'Shape error.'
        sys.exit(1)
        
of.write(struct.pack('i', total))
of.write(struct.pack('i', height))
of.write(struct.pack('i', width))

data = numpy.load(files[0])

for file in files[1:]:

    print 'Input file is ', file

    data = numpy.concatenate((data,numpy.load(file)))

    print 'data.ndim = ', data.ndim
    print 'data.shape = ', data.shape
    print 'data.size = ', data.size

numpy.random.shuffle(data)

data.astype('f').tofile(of)

of.close()

print 'All done.'
sys.exit(0)

