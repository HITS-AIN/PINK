#!/usr/bin/env python3

import getopt
import numpy
import matplotlib
import struct
import sys
import os.path
import math
import tools

class HeatmapVisualizer():
    def __init__(self, fileName):
        self.__fileName = fileName
        self.__numberOfImages = 0
        self.__somWidth = 0
        self.__somHeight = 0
        self.__somDepth = 0
        self.__maps = []

    def getNumberOfMaps(self):
        return self.__numberOfImages

    def getSomWidth(self):
        return self.__somWidth

    def getSomHeight(self):
        return self.__somHeight

    def getSomDepth(self):
        return self.__somDepth

    def getHeatmap(self, objectNumber):
        return self.__maps[objectNumber]

    #Reads in map data
    def readMap(self):
        #Unpacks the map parameters
        inputStream = open(self.__fileName, 'rb')
        tools.ignore_header_comments(inputStream)
        
        # <file format version> 2 <data-type> <number of entries> <som layout> <data>
        version, file_type, data_type, number_of_data_entries, self.__som_layout, som_dimensionality = struct.unpack('i' * 6, inputStream.read(4 * 6))
        print('version:', version)
        print('file_type:', file_type)
        print('data_type:', data_type)
        print('number_of_data_entries:', number_of_data_entries)
        print('som_layout:', self.__som_layout)
        print('som dimensionality:', som_dimensionality)
        som_dimensions = struct.unpack('i' * som_dimensionality, inputStream.read(4 * som_dimensionality))
        print('som dimensions:', som_dimensions)

        self.__numberOfImages = number_of_data_entries
        self.__somWidth = som_dimensions[0]
        self.__somHeight = som_dimensions[1] if som_dimensionality > 1 else 1
        self.__somDepth = som_dimensions[2] if som_dimensionality > 2 else 1

        print ("images: " + str(self.__numberOfImages))
        print ("width: " + str(self.__somWidth))
        print ("height: " + str(self.__somHeight))
        print ("depth: " + str(self.__somDepth))

        if self.__som_layout == 0: # box
            data = numpy.ones(self.__somWidth * self.__somHeight * self.__somDepth)
            for i in range(self.__somWidth * self.__somHeight * self.__somDepth):
                data[i] = struct.unpack_from("f", inputStream.read(4))[0]
            self.__maps.append(data)
        else: # hex
            data = numpy.ones(hexSize)
            for i in range(hexSize):
                data[i] = struct.unpack_from("f", inputStream.read(4))[0]
            self.__maps.append(data)

        self.__maps = numpy.array(self.__maps)
        print (str(len(self.__maps)) + " maps loaded")

    # Checks if hexagonal or cartesian map is used
    def isHexMap(self):
        return self.__som_layout == 1

    # Creates the SOM and saves it to the specified location. Displays map if --display, -d is set to 1.
    def showMap(self, imageNumber, neuronSize, shareIntensity = False, borderWidth = 2, facecolor = '#ffaadd'):
        print(shareIntensity)
        if facecolor != "#ffaadd":
            print ("WARNING! using non recommended background color! The results will look ugly.")
        figure = pyplot.figure(figsize=(16, 16))
        figure.patch.set_alpha(0.0)
        figure.patch.set_facecolor(facecolor)
        pyplot.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.01, top = 0.99, wspace = 0.1, hspace = 0.1)

        if self.__som_layout == 1:
            print ("hexagonal map")
            image = tools.calculate_map(self.__somWidth, self.__somHeight, self.__maps[imageNumber], neuronSize, neuronSize, shareIntensity=shareIntensity, border=borderWidth, shape="hex")
        else:
            print ("cartesian map")
            image = tools.calculate_map(self.__somWidth, self.__somHeight, self.__maps[imageNumber], neuronSize, neuronSize, shareIntensity=shareIntensity, border=borderWidth, shape="box")

        ax = pyplot.subplot()
        cmap = matplotlib.cm.get_cmap("jet")
        cmap.set_bad(facecolor,1.)
        #Uses image.T because the file is read in as C notation when it's actually Fortran notation.
        ax.imshow(image.T, aspect=1, interpolation="nearest", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        fileName = save+'/'+name+'%d.pdf' %imageNumber
        pyplot.savefig(fileName,bbox_inches='tight',dpi=150)
        if(display==1):
            pyplot.show()

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
    print ('  --neuron, -n <int>      Neuron size (default = 12).')
    print ('  --save, -s <String>     Location to save JPGs to.')
    print ('  --name, -o <String>     Name of saved file. Channel number will automatically be added to the end of the name.')
    print ('  --display, -d <int>     1 to display SOM as well as save it. Default 0.')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:i:b:n:s:o:d:",["help", "image=", "border=", "neuron=", "save=", "name=", "display="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    #Default parameters
    save="./"
    name="HEATMAP"
    borderWidth=2
    facecolor="#ffaadd"
    shareIntensity=True
    display=0
    imageNumber = 0
    neuronSize = 12

    #Set parameters
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-i", "--image"):
            imageNumber = int(arg)
        elif opt in ("-b", "--border"):
            borderWidth = int(arg)
        elif opt in ("-n", "--neuron"):
            neuronSize = int(arg)
        elif opt in ("-s", "--save"):
            save = arg
        elif opt in ("-o", "--name"):
            name = arg
        elif opt in ("-d", "--display"):
            display = int(arg)
     
    if len(args) != 1:
        print_usage()
        print ('ERROR: Input file is missing.')
        sys.exit(1)

    inputfile = args[0]

    print ('Input file is:', inputfile)
    print ('Save location is:', save)
    print ('The file name is:', name)
    print ('The border width is:', borderWidth)
    print ('The background color is:', facecolor)
    print ('The neuron size is:', neuronSize)
    if(display==1):
        print ('Display is on')
    else:
        print ('Display is off')
     
    #Importing here allows control of the backend depending on if display is on or off.
    if(display==0):
        matplotlib.use('Agg')
    from matplotlib import pyplot
   
    myVisualizer = HeatmapVisualizer(inputfile)
    myVisualizer.readMap()
    myVisualizer.showMap(imageNumber, neuronSize, shareIntensity, borderWidth, facecolor)























    
