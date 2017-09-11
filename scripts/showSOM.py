#!/usr/bin/python

from __future__ import print_function

import struct
import numpy
import matplotlib
#imports pyplot in code for control over backend
import math
import getopt
import sys

def print_usage():
    print ('')
    print ('Usage:')
    print ('')
    print ('  showSOM.py [Options] <inputfile>')
    print ('')
    print ('Options:')
    print ('')
    print ('  --help, -h                   Print this lines.')
    print ('  --save, -s <String>          Location to save JPGs to.')
    print ('  --intensity, -i <boolean>    Turn shareIntensity off.')
    print ('  --border, -b <int>           Border size.')
    print ('  --color, -c <String>         Background color. [Defaults to \\#ffaadd - HIGHLY RECOMMENDED!!!]')
    print ('  --name, -n <String>          Name of saved file. Channel number will automatically be added to the end of the name.')
    print ('  --display, -d <int>          1 to display SOM as well as save it. Default 0.')
    print ('')

class MAPVisualizer():
    def __init__(self, fileName):
        self.__fileName = fileName
        self.__numberOfChannels = 0
        self.__somWidth = 0
        self.__somHeight = 0
        self.__somDepth = 0
        self.__neuronWidth = 0
        self.__neuronHeight = 0
        self.__neurons = []

    #Returns number of channels
    def checkChannels(self):
        inputStream = open(self.__fileName, 'rb')
        self.__numberOfChannels = struct.unpack("i", inputStream.read(4))[0]
        return self.__numberOfChannels

    #Reads in map data
    def readMap(self):
        #Unpacks the map parameters
        inputStream = open(self.__fileName, 'rb')
        self.__numberOfChannels = struct.unpack("i", inputStream.read(4))[0]
        self.__somWidth = struct.unpack("i", inputStream.read(4))[0]
        self.__somHeight = struct.unpack("i", inputStream.read(4))[0]
        self.__somDepth = struct.unpack("i", inputStream.read(4))[0]
        self.__neuronWidth = struct.unpack("i", inputStream.read(4))[0]
        self.__neuronHeight = struct.unpack("i", inputStream.read(4))[0]

        print ("channels: " + str(self.__numberOfChannels))
        print ("width: " + str(self.__somWidth))
        print ("height: " + str(self.__somHeight))
        print ("depth: " + str(self.__somDepth))
        print ("neurons: " + str(self.__neuronWidth) +"x" + str(self.__neuronHeight))
        #Unpacks data
        try:
            while True:
                data = numpy.ones(self.__neuronWidth * self.__neuronHeight * self.__numberOfChannels)
                for i in range(self.__neuronWidth * self.__neuronHeight * self.__numberOfChannels):
                    data[i] = struct.unpack_from("f", inputStream.read(4))[0]
                self.__neurons.append(data)

        except:
            inputStream.close()
        self.__neurons = numpy.array(self.__neurons)

        print (str(len(self.__neurons)) + " neurons loaded")

    #Checks if hexagonal or quadratic map is used
    def isHexMap(self):
        return len(self.__neurons) < self.__somHeight * self.__somWidth

    def calculateMap(self, neurons, shareIntensity = False, border = 0, shape="box"):
        #For quadratic map, it reads through the data and creates each neuron as a 1D array and then resizes it to the neuronSize
        if shape == "box":
            neuronSize = numpy.array([self.__neuronWidth, self.__neuronHeight])
            mapSize = numpy.array([self.__somWidth, self.__somHeight])
            size = numpy.multiply(mapSize,numpy.array(neuronSize) + border) + border
            image = numpy.empty(size)
            image[:] = numpy.NAN
            for x in range(self.__somWidth):
                for y in range(self.__somHeight):
                    data = neurons[x + y*self.__somWidth].reshape(self.__neuronWidth, self.__neuronHeight)
                    if not shareIntensity:
                        if numpy.max(data)-numpy.min(data) != 0:
                            data = 1.0 * (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
                    image[x*(self.__neuronWidth + border): (x+1) * (self.__neuronWidth + border) -border, y * (self.__neuronHeight + border): (y+1) * (self.__neuronHeight + border) - border] = numpy.fliplr(data)
        #For hexagonal map, it goes through each neuron while accounting for the hexagonal shape, starting at the bottom left corner
        if shape == "hex":
            size = numpy.multiply((self.__somWidth+0.5,self.__somHeight),numpy.array((self.__neuronWidth, self.__neuronHeight)) + border) + border
            size[1] = math.ceil(size[1] - (self.__somHeight-1.0) * self.__neuronHeight / 4.0)
            size[0] = math.ceil(size[0])
            size = numpy.asarray(size, int)
            image = numpy.empty(size)
            image[:] = numpy.NAN
            mapY = 0
            mapX = abs((self.__somHeight-1)/2 - mapY)
            mapX = mapX / 2 + mapX % 2 - 1
            for neuron in neurons:
                mapX = mapX + 1
                off = abs((self.__somHeight-1)/2 - mapY)
                if mapX >= self.__somWidth - math.floor(off / 2) - off % 2 * (mapY) % 2:
                    mapY = mapY + 1
                    mapX = abs((self.__somHeight-1)/2 - mapY)
                    mapX = math.floor(mapX / 2) + mapX % 2 * (1-mapY) % 2
                if mapY >= self.__somHeight:
                    print("abort")
                    return image
                if not shareIntensity:
                    if numpy.max(neuron)-numpy.min(neuron) != 0:
                        neuron = 1.0 * (neuron - numpy.min(neuron)) / (numpy.max(neuron) - numpy.min(neuron))
                for xPos in range(self.__neuronWidth):
                    for yPos in range(self.__neuronHeight):
                        if math.floor(xPos/2.0 + yPos) < self.__neuronHeight / 4.0 or \
                           math.floor(xPos/2.0 + yPos + 1) > self.__neuronWidth + self.__neuronHeight / 4.0 or \
                           math.floor(xPos/2.0 - yPos + 1) > self.__neuronHeight / 4.0 or \
                           math.floor(xPos/2.0 - yPos) < -self.__neuronWidth + self.__neuronHeight / 4.0:
                            continue
                        else:
                            x = int(math.floor((1.0 * numpy.floor(mapX) + (mapY%2) / 2.0) * (self.__neuronWidth + border) + xPos + border))
                            y = int(math.floor(1.0 * mapY * (self.__neuronHeight * 3.0 / 4.0 + border) + yPos + border))
                            image[x][y] = neuron[xPos + yPos * self.__neuronWidth]
        return image

    #Creates the SOM and saves it to the specified location. Displays map if --display, -d is set to 1.
    def showMap(self, channel, shareIntensity = True, borderWidth = 2, facecolor = '#ffaadd'):
        if facecolor != "#ffaadd":
            print ("WARNING! using non recommended background color! The results will look ugly.")
        figure = pyplot.figure( figsize=(16, 16))
        figure.patch.set_alpha(0.0)
        figure.patch.set_facecolor(facecolor)
        pyplot.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.01, top = 0.99, wspace = 0.1, hspace = 0.1)
        start=int(len(self.__neurons[0]) / self.__numberOfChannels * channel)
        end=int(len(self.__neurons[0]) / self.__numberOfChannels * (channel+1))
        if self.isHexMap():
            print ("hexagonal map")
            image = self.calculateMap(self.__neurons[:,start:end], border=borderWidth, shareIntensity=shareIntensity, shape="hex")
        else:
            print ("quadratic map")
            image = self.calculateMap(self.__neurons[:,start:end], border=borderWidth, shareIntensity=shareIntensity, shape="box")

        ax = pyplot.subplot()
        if channel==0:
            cmap = matplotlib.cm.get_cmap("Blues")
        elif channel==1:
            cmap = matplotlib.cm.get_cmap("Reds")
        elif channel==2:
            cmap = matplotlib.cm.get_cmap("Greens")
        elif channel==3:
            cmap = matplotlib.cm.get_cmap("Purples")
        elif channel==4:
            cmap = matplotlib.cm.get_cmap("Greys")
        elif channel==5:
            cmap = matplotlib.cm.get_cmap("Oranges")
        else:
            cmap = matplotlib.cm.get_cmap()
        cmap.set_bad(facecolor,1.)
        ax.imshow(image.T, aspect='auto', interpolation="nearest", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        fileName = save+'/'+name+'%d.pdf' %channel
        pyplot.savefig(fileName)
        if(display==1):
            pyplot.show()


if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:s:b:c:i:n:d:",["help", "save=", "border=", "color=", "intensity=", "name=", "display="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    #Default parameters
    save=""
    name="SOM"
    borderWidth=2
    facecolor="#ffaadd"
    shareIntensity=True
    display=0

    #Use inputted parameters
    for opt, arg in opts:
        if opt in ("-s", "--save"):
            save = arg
        elif opt in ("-i", "--shareIntensity"):
            shareIntensity = arg
        elif opt in ("-c", "--color"):
            facecolor = arg
        elif opt in ("-b", "--border"):
            borderWidth = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-d", "--display"):
            display = int(arg)
        elif opt in ("-h", "--help"):
            print_usage()
            sys.exit()
            
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
    print ('The intensity is shared:', shareIntensity)
    if(display==1):
        print ('Display is on')
    else:
        print ('Display is off')
     
    #Importing here allows control of the backend depending on if display is on or off.
    if(display==0):
        matplotlib.use('Agg')
    from matplotlib import pyplot
   
    myVisualizer = MAPVisualizer(inputfile)
    myVisualizer.readMap()
    channels=myVisualizer.checkChannels()
    for i in range(0,channels):
        myVisualizer.showMap(i, shareIntensity, borderWidth, facecolor)