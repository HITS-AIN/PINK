#!/usr/bin/env python3

import struct
import numpy
import matplotlib
import math
import getopt
import sys
import tools

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

    def getNumberOfChannels(self):
        return self.__numberOfChannels

    def getSomWidth(self):
        return self.__somWidth

    def getSomHeight(self):
        return self.__somHeight

    def getSomDepth(self):
        return self.__somDepth

    #Reads in map data
    def readMap(self):
        #Unpacks the map parameters
        inputStream = open(self.__fileName, 'rb')
        tools.ignore_header_comments(inputStream)

        self.__somWidth, self.__somHeight, self.__somDepth, self.__neuronWidth, self.__neuronHeight, self.__numberOfChannels = struct.unpack('i' * 6, inputStream.read(4*6))

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
        inputStream.close()

    #Checks if hexagonal or quadratic map is used
    def isHexMap(self):
        return len(self.__neurons) < self.__somHeight * self.__somWidth * self.__somDepth

    #Creates the SOM and saves it to the specified location. Displays map if --display, -d is set to 1.
    def showMap(self, channel, shareIntensity = True, borderWidth = 2, facecolor = '#ffaadd'):
        print(shareIntensity)
        if facecolor != "#ffaadd":
            print ("WARNING! using non recommended background color! The results will look ugly.")
        figure = pyplot.figure(figsize=(16, 16))
        figure.patch.set_alpha(0.0)
        figure.patch.set_facecolor(facecolor)
        pyplot.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.01, top = 0.99, wspace = 0.1, hspace = 0.1)
        start=int(len(self.__neurons[0]) / self.__numberOfChannels * channel)
        end=int(len(self.__neurons[0]) / self.__numberOfChannels * (channel+1))
        if self.isHexMap():
            print ("hexagonal map")
            image = tools.calculate_map(self.__somWidth, self.__somHeight, self.__neurons[:,start:end], self.__neuronWidth, self.__neuronHeight, shareIntensity=shareIntensity, border=borderWidth, shape="hex")
        else:
            print ("quadratic map")
            image = tools.calculate_map(self.__somWidth, self.__somHeight, self.__neurons[:,start:end], self.__neuronWidth, self.__neuronHeight, shareIntensity=shareIntensity, border=borderWidth, shape="box")

        ax = pyplot.subplot()
        if somColor==0:
            cmap = matplotlib.cm.get_cmap("jet")
        else:
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
        #Uses image.T because the file is read in as C notation when it's actually Fortran notation.
        ax.imshow(image.T, aspect=1, interpolation="nearest", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        fileName = save+'/'+name+'%d.pdf' %channel
        pyplot.savefig(fileName,bbox_inches='tight',dpi=150)
        if(display==1):
            pyplot.show()

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
    print ('  --scolor, -o <int>           0 to use jet color scheme for all SOMs, 1 to use single color scheme with a different color per SOM. Default 1.')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:s:b:c:i:n:d:o:",["help", "save=", "border=", "color=", "intensity=", "name=", "display=","scolor="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    #Default parameters
    save="./"
    name="SOM"
    borderWidth=2
    facecolor="#ffaadd"
    shareIntensity=True
    display=0
    somColor=1

    #Use inputted parameters
    for opt, arg in opts:
        if opt in ("-s", "--save"):
            save = arg
        elif opt in ("-i", "--shareIntensity"):
            shareIntensity = arg in ["True", "true", "yes", "on", "yeah", "si", "wi", "ja"]
        elif opt in ("-c", "--color"):
            facecolor = arg
        elif opt in ("-b", "--border"):
            borderWidth = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-d", "--display"):
            display = int(arg)
        elif opt in ("-o", "--scolor"):
            somColor = int(arg)
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
    for i in range(0,myVisualizer.getNumberOfChannels()):
        myVisualizer.showMap(i, shareIntensity, borderWidth, facecolor)
