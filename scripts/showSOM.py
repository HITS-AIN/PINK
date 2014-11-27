#!/usr/bin/python

import getopt
import numpy
import matplotlib.pylab as plt
import struct
import sys

if __name__ == "__main__":

    inputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:c:",["ifile=", "channel="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile>'
        sys.exit(2)

    imageNumber = 0
    channelNumber = 0

    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-c", "--channel"):
            channelNumber = int(arg)

    print 'Input file is ', inputfile
    print 'Channel number is ', channelNumber

    inFile = open(inputfile, 'rb')
    size = struct.unpack('iii', inFile.read(12))

    print 'size = ', size

    numberOfChannels = size[0]

    if channelNumber >= numberOfChannels:
        print 'Channel number too large.'
        sys.exit(1)

    somDim = size[1]
    neuronDim = size[2]
    dataSize = numberOfChannels * somDim * somDim * neuronDim * neuronDim
    array = numpy.array(struct.unpack('f' * dataSize, inFile.read(dataSize * 4)))
    
    imageDim = somDim * neuronDim;
    data = numpy.ndarray([somDim, somDim, numberOfChannels, neuronDim, neuronDim], 'float', array)
    data = numpy.swapaxes(data, 1, 3)
    data = numpy.reshape(data, (imageDim, numberOfChannels, imageDim))

    inFile.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data[:,channelNumber,:], interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    print 'All done.'
    sys.exit()
