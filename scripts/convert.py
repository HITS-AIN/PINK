#!/usr/bin/python

import getopt
import Image
import matplotlib
import numpy
import sys
from matplotlib import pyplot

if __name__ == "__main__":

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    print 'Input file is ', inputfile
    print 'Output file is ', outputfile

    data = numpy.load(inputfile)
    
    #image = Image.fromarray(data, 'RGB')
    #image.show()

    ax = pyplot.subplot()
    ax.imshow(data[0], aspect='auto', cmap=matplotlib.cm.jet)

    print 'data.size = ', data.size
    print 'data[0].shape = ', data[0].shape

    data.astype('f').tofile(outputfile)

    print 'All done.'
    sys.exit()
