#!/usr/bin/python

import getopt
import numpy
import matplotlib.pylab as plt
import struct
import sys

if __name__ == "__main__":

    inputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:n:",["ifile=", "image=", "channel="])
    except getopt.GetoptError:
        print 'showHeatmap.py -i <inputfile>'
        sys.exit(2)

    imageNumber = 0

    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-n", "--image"):
            imageNumber = int(arg)

    print 'Input file is ', inputfile
    print 'Image number is ', imageNumber

    inFile = open(inputfile, 'rb')
    size = struct.unpack('iii', inFile.read(12))
    numberOfImages = size[0]
    somDimension = size[1]
    neuronDim = size[2]
    somSize = somDimension * somDimension

    print 'Number of images = ', numberOfImages 
    print 'SOM dimension = ', somDimension
    print 'Neuron dimension is ', neuronDim

    if imageNumber >= numberOfImages:
        print 'Image number too large.'
        sys.exit(1)

    inFile.seek(imageNumber * somSize * 4, 1)
    array = numpy.array(struct.unpack('f' * somSize, inFile.read(somSize * 4)))
    inFile.close()

    imageDim = somDimension * neuronDim
    image = numpy.ndarray(shape=(imageDim,imageDim), dtype=float, order='F')
    
    for i in range(somDimension):
        for j in range(somDimension):
            image[i*neuronDim:(i+1)*neuronDim, j*neuronDim:(j+1)*neuronDim] = array[i*somDimension+j]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(image, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()

    print 'All done.'
    sys.exit()
