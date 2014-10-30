#!/usr/bin/python

import getopt
import Image
import matplotlib
import numpy
import struct
import sys
from matplotlib import pyplot

def print_usage():
    print sys.argv[0], ' -o <outputfile> [inputfiles]'
    
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
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    
    files = args
    of = open(outputfile, 'wb')
    
    data = numpy.load(files[0])
    total = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    
    for file in files[1:]:
        data = numpy.load(file)
        total += data.shape[0]
        if (height != data.shape[1]):
            print 'Shape error.'
            sys.exit(1)
        if (width != data.shape[2]):
            print 'Shape error.'
            sys.exit(1)
            
    of.write(struct.pack('i', total))
    of.write(struct.pack('i', height))
    of.write(struct.pack('i', width))
    
    data = numpy.load(files[0])
    
    if shuffle:
        print 'Shuffle'
        for file in files[1:]:
        
            print 'Input file is ', file
        
            data = numpy.concatenate((data,numpy.load(file)))
        
            print 'data.ndim = ', data.ndim
            print 'data.shape = ', data.shape
            print 'data.size = ', data.size
        
        numpy.random.shuffle(data)
        
        data.astype('f').tofile(of)
    else:
        print 'No shuffle'
        for file in files[1:]:
        
            print 'Input file is ', file
        
            data = numpy.load(file)
        
            print 'data.ndim = ', data.ndim
            print 'data.shape = ', data.shape
            print 'data.size = ', data.size

            data.astype('f').tofile(of)

    of.close()
    
    print 'All done.'
    sys.exit(0)
    
