#!/usr/bin/env python3

import struct
import numpy
import matplotlib
import math
import getopt
import sys
import tools
from show_heatmap import HeatmapVisualizer as HV

class Rotations():
    def __init__(self, fileName):
        self.__fileName = fileName
        self.__numberOfImages = 0
        self.__somWidth = 0
        self.__somHeight = 0
        self.__somDepth = 0
        self.__flipped = []
        self.__rotations = []

    def getNumberOfImages(self):
        return self.__numberOfImages

    def getSomWidth(self):
        return self.__somWidth

    def getSomHeight(self):
        return self.__somHeight

    def getSomDepth(self):
        return self.__somDepth

    def getImageNeuron(self, objectNumber, neuron):
        return self.__flipped[:,objectNumber * self.__somWidth * self.__somHeight * self.__somDepth + neuron], self.__rotations[:,objectNumber * self.__somWidth * self.__somHeight * self.__somDepth + neuron]

    def getImageBestNeuron(self, objectNumber, mapping):
        mv = HV(mapping)
        mv.readMap()
        bestNeuron = numpy.argmin(mv.getHeatmap(objectNumber))
        return self.getImageNeuron(objectNumber, bestNeuron)

    def getAllImages(self, mapping):
        bestRot = numpy.ones((self.__numberOfImages,2))
        mv = HV(mapping)
        mv.readMap()
        for i in range(self.__numberOfImages):
            bestNeuron = numpy.argmin(mv.getHeatmap(i))
            bestRot[i] = self.getImageNeuron(i, bestNeuron)
        return bestRot

    #Reads in map data
    def readRotations(self):
        #Unpacks the map parameters
        inputStream = open(self.__fileName, 'rb')
        tools.ignore_header_comments(inputStream)
        
        # <file format version> 3 <number of entries> <som layout> <data>
        version, file_type, number_of_data_entries, som_layout, som_dimensionality = struct.unpack('i' * 5, inputStream.read(4 * 5))
        print('version:', version)
        print('file_type:', file_type)
        print('number_of_data_entries:', number_of_data_entries)
        print('som_layout:', som_layout)
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

        #Unpacks data
        try:
            while True:
                dataF = numpy.ones(self.__somWidth * self.__somHeight * self.__somDepth * self.__numberOfImages)
                dataR = numpy.ones(self.__somWidth * self.__somHeight * self.__somDepth * self.__numberOfImages)
                for i in range(self.__somWidth * self.__somHeight * self.__somDepth * self.__numberOfImages):
                    dataF[i] = struct.unpack_from("?", inputStream.read(1))[0]
                    dataR[i] = struct.unpack_from("f", inputStream.read(4))[0]
                self.__flipped.append(dataF)
                self.__rotations.append(dataR)

        except:
            inputStream.close()

        self.__flipped = numpy.array(self.__flipped)
        self.__rotations = numpy.array(self.__rotations)

        print ("rotations loaded")
        inputStream.close()


def print_usage():
    print ('')
    print ('Usage:')
    print ('')
    print ('  rotations.py [Options] <inputfile>')
    print ('')
    print ('Options:')
    print ('')
    print ('  --help, -h                   Print this lines.')
    print ('  --image, -i <int>            Image number.')
    print ('  --mapping, -m <String>       Mapping file for finding best neurons.')
    print ('')

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"h:i:m:",["help", "image=", "mapping="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    # Default parameters
    image = 0
    mapping = ""

    # Use input parameters
    for opt, arg in opts:
        if opt in ("-i", "--image"):
            image = int(arg)
        elif opt in ("-m", "--mapping"):
            mapping = arg
        elif opt in ("-h", "--help"):
            print_usage()
            sys.exit()
            
    if len(args) != 1:
        print_usage()
        print ('ERROR: Input file is missing.')
        sys.exit(1)

    inputfile = args[0]

    print ('Input file is:', inputfile)
    print ('The image number is:', image)
    print ('The mapping file is:', mapping)

    myRotations = Rotations(inputfile)
    myRotations.readRotations()

    if image==-1:
        print(myRotations.getAllImages(mapping))
    else:
        print(myRotations.getImageBestNeuron(image,mapping))
