#!/usr/bin/env python3

import getopt
import numpy
import matplotlib
import struct
import sys
import tools

def print_usage():
    print ('')
    print ('Usage:')
    print ('')
    print ('  showImages.py [Options] <inputfile>')
    print ('')
    print ('Options:')
    print ('')
    print ('  --channel, -c <int>     Number of channel to visualize (default = 0).')
    print ('  --help, -h              Print this lines.')
    print ('  --image, -i <int>       Number of image to visualize (default = 0).')
    print ('  --save, -s <String>     Location to save JPGs to.')
    print ('  --name, -n <String>     Name of saved file.')
    print ('  --display, -d <int>     1 to display SOM as well as save it. Default 0.')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:n:c:n:s:d:",["help", "image=", "channel=", "name=", "save=", "display="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    imageNumber = 0
    channelNumber = 0
    save = "./"
    name= "image"
    display = 0

    for opt, arg in opts:
        if opt in ("-c", "--channel"):
            channelNumber = int(arg)
        elif opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-i", "--image"):
            imageNumber = int(arg)
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
    print ('Channel number is ', channelNumber)
    
    if(display==1):
        print ('Display is on')
    else:
        print ('Display is off')

    #Importing here allows control of the backend depending on if display is on or off.
    if(display==0):
        matplotlib.use('Agg')
    from matplotlib import pyplot

    file = open(inputfile, 'rb')
    tools.ignore_header_comments(file)
    
    # <file format version> 0 <data-type> <number of entries> <data layout> <data>
    version, file_type, data_type, numberOfImages, layout, dimensionality = struct.unpack('i' * 6, file.read(4 * 6))
    print('version:', version)
    print('file_type:', file_type)
    print('data_type:', data_type)
    print('numberOfImages:', numberOfImages)
    print('layout:', layout)
    print('dimensionality:', dimensionality)

    numberOfChannels = struct.unpack('i', file.read(4))[0] if dimensionality > 2 else 1
    height = struct.unpack('i', file.read(4))[0] if dimensionality > 1 else 1
    width = struct.unpack('i', file.read(4))[0]

    print('width:', width)
    print('height:', height)
    print('numberOfChannels:', numberOfChannels)

    if imageNumber >= numberOfImages:
        print ('Image number too large.')
        sys.exit(1)

    if channelNumber >= numberOfChannels:
        print ('Channel number too large.')
        sys.exit(1)

    size = width * height
    file.seek((imageNumber*numberOfChannels + channelNumber) * size*4, 1)
    array = numpy.array(struct.unpack('f' * size, file.read(size*4)))
    data = numpy.ndarray([width,height], 'float', array)

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    pyplot.imshow(data, interpolation='nearest', cmap=pyplot.cm.jet)
    pyplot.colorbar()
    fileName = save+'/'+name+'.pdf'
    pyplot.savefig(fileName)
    if(display==1):
        pyplot.show()

    print ('All done.')
    sys.exit()
