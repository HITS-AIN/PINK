#!/usr/bin/python

import getopt
import numpy
import matplotlib.pylab as plt
import struct
import sys

def print_usage():
    print 'showImages.py [Options] <inputfile>'
    print ''
    print 'Options:'
    print ''
    print '  --channel, -c   Number of channel to visualize (default = 0).'
    print '  --help, -h      Print this lines.'
    print '  --image, -i     Number of image to visualize (default = 0).'

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:n:c:",["help", "image=", "channel="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    imageNumber = 0
    channelNumber = 0

    for opt, arg in opts:
        if opt in ("-c", "--channel"):
            channelNumber = int(arg)
        elif opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-i", "--image"):
            imageNumber = int(arg)

    if len(args) != 1:
        print_usage()
        print 'ERROR: Input file is missing.'
        sys.exit(1)

    inputfile = args[0]

    print 'Input file is ', inputfile
    print 'Image number is ', imageNumber
    print 'Channel number is ', channelNumber

    inFile = open(inputfile, 'rb')
    numberOfImages, numberOfChannels, width, heigth = struct.unpack('i' * 4, inFile.read(4 * 4))

    print 'Number of images = ', numberOfImages
    print 'Number of channels = ', numberOfChannels
    print 'Width = ', width
    print 'Heigth = ', heigth

    if imageNumber >= numberOfImages:
        print 'Image number too large.'
        sys.exit(1)

    if channelNumber >= numberOfChannels:
        print 'Channel number too large.'
        sys.exit(1)

    size = width * heigth
    inFile.seek((imageNumber*numberOfChannels + channelNumber) * size*4, 1)
    array = numpy.array(struct.unpack('f' * size, inFile.read(size*4)))
    data = numpy.ndarray([width,heigth], 'float', array)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    print 'All done.'
    sys.exit()
