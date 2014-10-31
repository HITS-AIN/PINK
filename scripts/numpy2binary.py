import getopt
import matplotlib
import math
import numpy
import struct
import sys
from matplotlib import pyplot

def print_usage():
    print '<command> -o <outputfile> [inputfiles]'
    
if __name__ == "__main__":
    
    shuffle = False
    outputfile = 'result.bin'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:s",["ofile="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt == "-s":
            shuffle = True
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    
    files = args
    of = open(outputfile, 'wb')

    total = 0
    max_height = 0
    min_height = sys.maxint
    max_width = 0
    min_width = sys.maxint

    for file in files:
        data = numpy.load(file)
        total += data.shape[0]
        for i in xrange(len(data)):
            if (data[i].shape[0] > max_height):
                max_height = data[i].shape[0]
            if (data[i].shape[0] < min_height):
                min_height = data[i].shape[0]
            if (data[i].shape[1] > max_width):
                max_width = data[i].shape[1]
            if (data[i].shape[1] < min_width):
                min_width = data[i].shape[1]

    print 'number = ', total
    print 'max_height = ', max_height
    print 'min_height = ', min_height
    print 'max_width  = ', max_width
    print 'min_width  = ', min_width

    dim = min(min_height,min_width)
    print 'dim = ', dim

    of.write(struct.pack('i', total))
    of.write(struct.pack('i', dim))
    of.write(struct.pack('i', dim))
    
    for file in files:
    
        print 'Input file is ', file
    
        data = numpy.load(file)

        for i in xrange(len(data)):

            image = data[i]
            image = 1.0 * image / numpy.max(image)
            indices = numpy.where(image<numpy.std(image)*3.0)
            image[indices] = 0

            if numpy.isnan(image.any()):
                print 'image entry is nan.'
                sys.exit(1)

            if (image.shape[0] != dim or image.shape[1] != dim):
                image[(image.shape[0]-dim)/2:(image.shape[0]+dim)/2, (image.shape[1]-dim)/2:(image.shape[1]+dim)/2].astype('f').tofile(of)
            else:
                image.astype('f').tofile(of)

    of.close()
    
    print 'All done.'
    sys.exit(0)

