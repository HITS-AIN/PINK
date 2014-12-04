#!/usr/bin/python

import getopt
import numpy
import matplotlib.pylab as plt
import struct
import sys

def print_usage():
    print ''
    print 'Usage:'
    print ''
    print '  showSOM.py [Options] <inputfile>'
    print ''
    print 'Options:'
    print ''
    print '  --channel, -c <int>      Number of channel to visualize (default = 0).'
    print '  --help, -h               Print this lines.'
    print ''

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hc:",["help", "channel="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    imageNumber = 0
    channelNumber = 0

    for opt, arg in opts:
        if opt in ("-c", "--channel"):
            channelNumber = int(arg)
        elif opt in ("-h", "--help"):
            print_usage()
            sys.exit()

    if len(args) != 1:
        print_usage()
        print 'ERROR: Input file is missing.'
        sys.exit(1)

    inputfile = args[0]

    print 'Input file is ', inputfile
    print 'Channel number is ', channelNumber

    inFile = open(inputfile, 'rb')
    numberOfChannels, SOM_width, SOM_height, neuron_width, neuron_height = struct.unpack('i' * 5, inFile.read(4*5))

    print 'Number of channels = ', numberOfChannels
    print 'SOM_width = ', SOM_width
    print 'SOM_height = ', SOM_height
    print 'neuron_width = ', neuron_width
    print 'neuron_height = ', neuron_height

    if channelNumber >= numberOfChannels:
        print 'Channel number too large.'
        sys.exit(1)

    dataSize = numberOfChannels * SOM_width * SOM_height * neuron_width * neuron_height
    array = numpy.array(struct.unpack('f' * dataSize, inFile.read(dataSize * 4)))
    
    image_width = SOM_width * neuron_width;
    image_height = SOM_height * neuron_height;
    data = numpy.ndarray([SOM_width, SOM_height, numberOfChannels, neuron_width, neuron_height], 'float', array)
    data = numpy.swapaxes(data, 1, 3)
    data = numpy.reshape(data, (image_width, numberOfChannels, image_height))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data[:,channelNumber,:], interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    print 'All done.'
    sys.exit()
