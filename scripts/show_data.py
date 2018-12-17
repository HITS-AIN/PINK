#!/usr/bin/env python3

"""
PINK Visualization of binary data files
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import argparse
import numpy
import matplotlib
import struct
import tools

def main():
    """ Main routine of PINK Visualization of binary data files """

    parser = argparse.ArgumentParser(description='PINK convert binary formats')
    parser.add_argument('data', help='Data input file (.bin)', action=tools.check_extension({'bin'}))
    parser.add_argument('-o', '--output', help='Data output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be talkative')
    args = parser.parse_args()
    
    print('Input file is ', inputfile)
    print('Image number is ', imageNumber)
    print('Channel number is ', channelNumber)
    
    if(display==1):
        print ('Display is on')
    else:
        print ('Display is off')

    #Importing here allows control of the backend depending on if display is on or off.
    if(display==0):
        matplotlib.use('Agg')
    from matplotlib import pyplot

    file = open(inputfile, 'rb')
        
    last_position = file.tell()
    for line in file:
        if line[:1] != b'#':
            break
        last_position = file.tell()
     
    file.seek(last_position, 0)
    numberOfImages, numberOfChannels, width, height = struct.unpack('i' * 4, file.read(4 * 4))

    print ('Number of images = ', numberOfImages)
    print ('Number of channels = ', numberOfChannels)
    print ('Width = ', width)
    print ('Height = ', height)

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

main()
