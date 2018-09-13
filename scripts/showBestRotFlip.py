#!/usr/bin/python3

import getopt
import numpy
import matplotlib
import struct
import sys

def print_usage():
    print ('')
    print ('Usage:')
    print ('')
    print ('  showHeatmap.py [Options] <inputfile>')
    print ('')
    print ('Options:')
    print ('')
    print ('  --help, -h              Print this lines.')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h",["help"])
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

    if len(args) != 1:
        print_usage()
        print ('ERROR: Input file is missing.')
        sys.exit(1)

    inputfile = args[0]

    print ('Input file is ', inputfile)

    inFile = open(inputfile, 'rb')
    numberOfImages, SOM_width, SOM_height, SOM_depth = struct.unpack('i' * 4, inFile.read(4*4))

    print ('Number of images = ', numberOfImages)
    print ('SOM_width = ', SOM_width)
    print ('SOM_height = ', SOM_height)
    print ('SOM_depth = ', SOM_depth)
    
    size = SOM_width * SOM_height * SOM_depth
    data = numpy.fromfile(inFile, dtype = numpy.dtype('if'), count = size)
    print (data)
   
    #array = numpy.array(struct.unpack('if', inFile.read(size * 2 * 4))
    #print (array)

    print ('All done.')
    sys.exit()
