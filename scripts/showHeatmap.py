#!/usr/bin/python

from __future__ import print_function

import getopt
import numpy
import matplotlib
#imports pyplot in code for control over backend
import struct
import sys
import os.path
import math
import somTools

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
        somTools.ignoreHeaderComments(inputStream) # find end of header

        self.__numberOfImages = struct.unpack("i", inputStream.read(4))[0]
        self.__somWidth = struct.unpack("i", inputStream.read(4))[0]
        self.__somHeight = struct.unpack("i", inputStream.read(4))[0]
        self.__somDepth = struct.unpack("i", inputStream.read(4))[0]

        print ("images: " + str(self.__numberOfImages))
        print ("width: " + str(self.__somWidth))
        print ("height: " + str(self.__somHeight))
        print ("depth: " + str(self.__somDepth))

        start = inputStream.tell()      
        if os.path.getsize(self.__fileName) < self.__numberOfImages * self.__somWidth * self.__somHeight * self.__somDepth * 4 + start:
            self.__shape = "hex"
        else:
            self.__shape = "box"

        #Unpacks data
        hexSize=int(1.0 + 6.0 * (( (self.__somWidth-1)/2.0 + 1.0) * (self.__somHeight-1)/ 4.0))
        try:
            while True:
                if self.__shape == "box":
                    data = numpy.ones(self.__somWidth * self.__somHeight * self.__somDepth)
                    for i in range(self.__somWidth * self.__somHeight * self.__somDepth):
                        data[i] = struct.unpack_from("f", inputStream.read(4))[0]
                    self.__maps.append(data)
                else:
                    
                    data = numpy.ones(hexSize)
                    for i in range(hexSize):
                        data[i] = struct.unpack_from("f", inputStream.read(4))[0]
                    self.__maps.append(data)

        except:
            inputStream.close()
        self.__maps = numpy.array(self.__maps)
        print (str(len(self.__maps)) + " maps loaded")

    #Checks if hexagonal or quadratic map is used
    def isHexMap(self):
        return self.__shape == "hex"

    #Creates the SOM and saves it to the specified location. Displays map if --display, -d is set to 1.
    def showMap(self, imageNumber, neuronSize, shareIntensity = False, borderWidth = 2, facecolor = '#ffaadd'):
        print(shareIntensity)
        if facecolor != "#ffaadd":
            print ("WARNING! using non recommended background color! The results will look ugly.")
        figure = pyplot.figure(figsize=(16, 16))
        figure.patch.set_alpha(0.0)
        figure.patch.set_facecolor(facecolor)
        pyplot.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.01, top = 0.99, wspace = 0.1, hspace = 0.1)

        if self.isHexMap():
            print ("hexagonal map")
            image = somTools.calculateMap(self.__somWidth, self.__somHeight, self.__maps[imageNumber], neuronSize, neuronSize, shareIntensity=shareIntensity, border=borderWidth, shape="hex")
        else:
            print ("quadratic map")
            image = somTools.calculateMap(self.__somWidth, self.__somHeight, self.__maps[imageNumber], neuronSize, neuronSize, shareIntensity=shareIntensity, border=borderWidth, shape="box")

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























    
