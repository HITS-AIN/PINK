#!/usr/bin/python

from __future__ import print_function

import getopt
import numpy
import matplotlib
#imports pyplot in code for control over backend
import struct
import sys
from math import ceil, floor

def print_usage():
    print ('')
    print ('Usage:')
    print ('')
    print ('  showHeatmap.py [Options] <inputfile>')
    print ('')
    print ('Options:')
    print ('')
    print ('  --help, -h              Print this lines.')
    print ('  --image, -i <int>       Number of image to visualize (default = 0).')
    print ('  --border, -b <int>      Border size.')
    print ('  --neuron, -n <int>      Neuron size.')
    print ('  --save, -s <String>     Location to save JPGs to.')
    print ('  --name, -n <String>     Name of saved file. Channel number will automatically be added to the end of the name.')
    print ('  --display, -d <int>     1 to display SOM as well as save it. Default 0.')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:i:b:n:s:n:d:",["help", "image=", "border=", "neuron=", "save=", "name=", "display="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    #Default parameters
    imageNumber = 0
    channelNumber = 0
    border = 1
    neuronSize = 5
    save = ""
    name = "heatmap"
    display = 0

    #Set parameters
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-i", "--image"):
            imageNumber = int(arg)
        elif opt in ("-b", "--border"):
            border = int(arg)
        elif opt in ("-n", "--neuron"):
            neuronSize = int(arg)
        elif opt in ("-s", "--save"):
            save = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-d", "--display"):
            display = int(arg)

    if len(args) != 1:
        print_usage()
        print ('ERROR: Input file is missing.')
        sys.exit(1)

    inputfile = args[0]

    print ('Input file is ', inputfile)
    print ('Image number is ', imageNumber)
    print ('Border size is ', border)
    print ('Neuron size is ', neuronSize)
    if(display==1):
        print ('Display is on')
    else:
        print ('Display is off')

    #Importing here allows control of the backend depending on if display is on or off.
    if(display==0):
        matplotlib.use('Agg')
    import matplotlib.pylab as plt

    inFile = open(inputfile, 'rb')
    numberOfImages, SOM_width, SOM_height, SOM_depth = struct.unpack('i' * 4, inFile.read(4*4))

    print ('Number of images = ', numberOfImages)
    print ('SOM_width = ', SOM_width)
    print ('SOM_height = ', SOM_height)
    print ('SOM_depth = ', SOM_depth)

    if imageNumber >= numberOfImages:
        print ('Image number too large.')
        sys.exit(1)

    #Size if the map is hexagonal
    hexSize=int(1.0 + 6.0 * (( (SOM_width-1)/2.0 + 1.0) * (SOM_width-1)/ 4.0))
    #Size if the map is quadratic
    size = SOM_width * SOM_height * SOM_depth
    image_width = SOM_width
    image_height = SOM_depth * SOM_height
    inFile.seek(imageNumber * size * 4, 1)
    array = numpy.array(struct.unpack('f' * size, inFile.read(size * 4)))
    data = numpy.ndarray([SOM_width, SOM_height, SOM_depth], 'float', array)

    #Checks if the map is hexagonal. If hexagonal, all neurons between hexSize and size will be identical
    if numpy.min(data.flatten()[hexSize:]) - numpy.max(data.flatten()[hexSize:]) == 0:
        print ("hex heatmap")
        #Sets size to hexSize, including borders
        size = [int(ceil(SOM_width * ((neuronSize*2+1) + border) + neuronSize/2.0 + 1)),
                int(SOM_height * (neuronSize + neuronSize/2.0 + 1 + border) )]
        image = numpy.empty(size)
        image[:] = numpy.NAN
        data = data.reshape(SOM_width*SOM_height, -1)

        #Creates each neuron and fills it with the same color
        def fillHexagon(x, y, value):
            for yPos in range(-neuronSize, neuronSize+1):
                if yPos <= -neuronSize / 2.0:
                    xs = range(-(neuronSize+yPos)*2, (neuronSize+yPos)*2+1)
                elif yPos > neuronSize / 2.0:
                    xs = range(-(neuronSize-yPos)*2, (neuronSize-yPos)*2+1)
                else:
                    xs = range(-neuronSize, neuronSize+1)
                for xPos in xs:
                    image[int((x + y%2/2.0) * ((neuronSize*2+1) + border) + xPos + neuronSize/2.0),
                          int(y * ((neuronSize+floor(neuronSize/2.0)+1) + border) + yPos + neuronSize + 1)] = value

        y = 0
        x = int(SOM_width / 4.0 + 1)
        for part in data[:hexSize]:
            #Fills in the "center box" of each mini hexagon
            fillHexagon(x, y, part)
            x = x + 1
            #Fills in the "top triangle"
            if y < floor(SOM_height / 2.0) and (x>SOM_width - (floor(SOM_width / 4.0) + 1 - floor(y/2.0))):
                y = y + 1
                x = int(floor(SOM_width / 4.0) + 1 - ceil(y/2.0))
            #Fills in the "bottom triangle"
            elif (x>SOM_width - floor( (y-floor(SOM_height/2.0)) /2.0) - 1 ):
                y = y + 1
                x = int(ceil( (y-floor(SOM_height/2.0)) /2.0))
        #Transposes the data
        data = image.T
    else:
        #If the map is quadratic, swapping the axes and then transposing the data creates the heat map
        print ("box heatmap")
        data = numpy.swapaxes(data, 0, 2)
        data = numpy.reshape(data, (image_height, image_width)).T

    #Saves the heat map
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.colorbar()
    fileName = save+'/'+name+'.pdf'
    plt.savefig(fileName)
    if(display==1):
        plt.show()

    print ('All done.')
    sys.exit()
