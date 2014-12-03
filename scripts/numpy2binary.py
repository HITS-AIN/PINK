import getopt
import matplotlib
import math
import numpy
import struct
import sys
from matplotlib import pyplot

def print_usage():
    print ''
    print 'Usage:'
    print ''
    print '  numpy2binary.py [Options] <inputfile>'
    print ''
    print 'Options:'
    print ''
    print '  --broken, -b <string>     Behavior for images containing NaN (Skip = default, SetToZero).'
    print '  --help, -h                Print this lines.'
    print '  --norm, -n                Normalize image data.'
    print '  --ofile, -o <string>      Filename for the converted images (default = result.bin).'
    print ''
    
if __name__ == "__main__":

    norm = False
    outputfile = 'result.bin'
    BrokenImageBehavior = 'Skip'

    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:nb:",["help", "ofile=", "norm", "broken="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-n", "--norm"):
            norm = True
        elif opt in ("-b", "--broken"):
            if arg not in ['Skip', 'SetToZero']:
                print 'Unkown option for broken ', arg
                sys.exit(1)
            BrokenImageBehavior = arg

    if len(args) == 0:
        print_usage()
        sys.exit(1)

    files = args
    of = open(outputfile, 'wb')

    numberOfImages = 0
    numberOfChannels = 1
    nbBrokenImages = 0
    max_height = 0
    min_height = sys.maxint
    max_width = 0
    min_width = sys.maxint

    for file in files:
        data = numpy.load(file)
        for i in xrange(len(data)):

            image = data[i]

            if numpy.isnan(image).any():
                nbBrokenImages += 1
                if BrokenImageBehavior == 'Skip':
                    continue

            if (data[i].shape[0] > max_height):
                max_height = data[i].shape[0]
            if (data[i].shape[0] < min_height):
                min_height = data[i].shape[0]
            if (data[i].shape[1] > max_width):
                max_width = data[i].shape[1]
            if (data[i].shape[1] < min_width):
                min_width = data[i].shape[1]

            numberOfImages += 1

    print 'Number of images = ', numberOfImages
    print 'Number of images containing NaN elements = ', nbBrokenImages
    print 'Number of channels = ', numberOfChannels
    print 'max_height = ', max_height
    print 'min_height = ', min_height
    print 'max_width  = ', max_width
    print 'min_width  = ', min_width

    dim = min(min_height,min_width)
    print 'dim = ', dim

    of.write(struct.pack('i', numberOfImages))
    of.write(struct.pack('i', numberOfChannels))
    of.write(struct.pack('i', dim))
    of.write(struct.pack('i', dim))
    
    for file in files:
    
        print 'Input file is ', file
    
        data = numpy.load(file)

        for i in xrange(len(data)):

            image = data[i]

            if numpy.isnan(image).any():
                if BrokenImageBehavior == 'Skip':
                    continue
                elif BrokenImageBehavior == 'SetToZero':
                    image = numpy.nan_to_num(image)

            if norm:
                image = 1.0 * image / numpy.max(image)
                indices = numpy.where(image < numpy.std(image)*3.0)
                image[indices] = 0

            if (image.shape[0] != dim or image.shape[1] != dim):
                image[(image.shape[0]-dim)/2:(image.shape[0]+dim)/2, (image.shape[1]-dim)/2:(image.shape[1]+dim)/2].astype('f').tofile(of)
            else:
                image.astype('f').tofile(of)

    of.close()

    print 'All done.'
    sys.exit(0)

