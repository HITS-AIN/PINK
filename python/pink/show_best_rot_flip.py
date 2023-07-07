#!/usr/bin/env python3

import getopt
import numpy as np
import os
import struct
import sys

def print_usage():
    print ('')
    print ('Usage:')
    print ('')
    print ('  show_best_rot_flip.py [Options] <inputfile>')
    print ('')
    print ('Options:')
    print ('')
    print ('  --help, -h              Print this lines.')
    print ('  --image, -i <int>       Number of image to visualize (default = 0).')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:i:",["help", "image="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    #Default parameters
    image_number = 0

    #Set parameters
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-i", "--image"):
            image_number = int(arg)

    if len(args) != 1:
        print_usage()
        print ('ERROR: Input file is missing.')
        sys.exit(1)

    inputfile = args[0]

    print ('Input file is', inputfile)
    print ('Image number =', image_number)    
    
    filesize = os.path.getsize(inputfile)
    file = open(inputfile, 'rb')
    number_of_images, som_width, som_height, som_depth = struct.unpack('i' * 4, file.read(4*4))

    print ('Number of images =', number_of_images)
    print ('som_width =', som_width)
    print ('som_height =', som_height)
    print ('som_depth =', som_depth)
    
    radius = int((som_width - 1) / 2)
    size_header = 16
    size_pixel = 5
    if (filesize - size_header) / size_pixel / number_of_images == som_width * som_height * som_depth:
        layout = 'cartesian'
        number_of_neurons = som_width * som_height * som_depth
        
    elif (filesize - size_header) / size_pixel / number_of_images == som_width * som_width - radius * (radius + 1):
        layout = 'hexagonal'
        number_of_neurons = som_width * som_width - radius * (radius + 1)
        
    else:
        print ('ERROR: Can\'t detect layout.')
        sys.exit(1)

    print('layout =', layout)
    print('number_of_neurons =', number_of_neurons)

    file.seek(image_number * number_of_neurons * size_pixel, 1)
    data = np.fromfile(file, dtype = np.dtype([('flip', np.int8), ('angle', np.float32)]), count = number_of_neurons)
    print (data)

    print ('All done.')
    sys.exit()
