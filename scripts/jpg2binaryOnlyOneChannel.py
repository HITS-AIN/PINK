import getopt
import matplotlib
import math
import struct
import sys
from matplotlib import pyplot
from scipy import misc
import numpy as np

def print_usage():
    print '<command> -o <outputfile> [inputfiles]'
    
if __name__ == "__main__":

    norm = False
    outputfile = 'result.bin'
    channel = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:c:",["ofile=", "channel="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-c", "--channel"):
            channel = arg

    if len(args) == 0:
        print_usage()
        sys.exit(1)

    files = args
    of = open(outputfile, 'wb')

    max_height = 0
    min_height = sys.maxint
    max_width = 0
    min_width = sys.maxint

    for file in files:
    
        image = misc.imread(file)

        if (image.shape[0] > max_height):
            max_height = image.shape[0]
        if (image.shape[0] < min_height):
            min_height = image.shape[0]
        if (image.shape[1] > max_width):
            max_width = image.shape[1]
        if (image.shape[1] < min_width):
            min_width = image.shape[1]

    print 'Number of images = ', len(files)
    print 'max_height = ', max_height
    print 'min_height = ', min_height
    print 'max_width  = ', max_width
    print 'min_width  = ', min_width

    dim = min(min_height,min_width)
    print 'dim = ', dim

    of.write(struct.pack('i', len(files)))
    of.write(struct.pack('i', 1))
    of.write(struct.pack('i', dim))
    of.write(struct.pack('i', dim))

    for file in files:

        print 'Input file is ', file

        image = misc.imread(file)
        print image.shape

        if (image.shape[0] != dim or image.shape[1] != dim):
            image[(image.shape[0]-dim)/2:(image.shape[0]+dim)/2, (image.shape[1]-dim)/2:(image.shape[1]+dim)/2, channel].astype('f').tofile(of)
        else:
            image[:, :, channel].astype('f').tofile(of)

    of.close()

    print 'All done.'
    sys.exit(0)

