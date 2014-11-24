#!/usr/bin/python

import getopt
import numpy
import matplotlib.pylab as plt
import struct
import sys

if __name__ == "__main__":

    inputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:n:c:",["ifile=", "image=", "channel="])
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
        elif opt in ("-n", "--image"):
            imageNumber = int(arg)
        elif opt in ("-c", "--channel"):
            channelNumber = int(arg)

    print 'Input file is ', inputfile
    print 'Image number is ', imageNumber
    print 'Channel number is ', channelNumber

    inFile = open(inputfile, 'rb')
    size = struct.unpack('iiii', inFile.read(16))

    print 'size = ', size
    print 'size = ', size[1]
    
    if imageNumber > size[0]:
        print 'Image number too large.'
        sys.exit(1)

    if channelNumber > size[1]:
        print 'Channel number too large.'
        sys.exit(1)

    inFile.seek((imageNumber*size[1] + channelNumber) * size[2]*size[3]*4, 1)
    array = numpy.array(struct.unpack('f'*size[2]*size[3], inFile.read(size[2]*size[3]*4)))
    data = numpy.ndarray([size[2],size[3]], 'float', array)

    inFile.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    print 'All done.'
    sys.exit()
