"""
PINK common functions
"""

import argparse
import numpy as np
import math
import os
import struct

def check_extension(choices):
    """ Check in argparse if file extensions are supported """
    
    class Act(argparse.Action):
        def __call__(self,parser,namespace,fname,option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("file doesn't end with one of {}{}".format(choices,option_string))
            else:
                setattr(namespace,self.dest,fname)

    return Act


def load_data(filename):
    """ Load data from binary file """
    
    file = open(filename, 'rb')
    ignore_header_comments(file)
    
    numberOfImages, numberOfChannels, width, height = struct.unpack('i' * 4, file.read(4 * 4))

    size = numberOfImages * numberOfChannels * width * height
    array = np.array(struct.unpack('f' * size, file.read(size*4)))

    return np.ndarray([numberOfImages, numberOfChannels, width, height], 'float', array)


def save_data(filename, data):
    """ Write data as binary file """
    
    file = open(filename, 'wb')
    
    # Add channels
    if len(data.shape) == 3:
        print('Add channels')
        data = np.expand_dims(data, axis=1)
    
    print('shape', np.shape(data))
    print('numberOfImages', np.shape(data)[0])
    print('numberOfChannels', np.shape(data)[1])
    print('width', np.shape(data)[2])
    print('height', np.shape(data)[3])
    
    file.write(struct.pack('i', np.shape(data)[0]))
    file.write(struct.pack('i', np.shape(data)[1]))
    file.write(struct.pack('i', np.shape(data)[2]))
    file.write(struct.pack('i', np.shape(data)[3]))
    file.write(struct.pack('f' * np.shape(data)[0] * np.shape(data)[1] * np.shape(data)[2] * np.shape(data)[3], *data.flatten()))

    data.tofile(file)


def ignore_header_comments(inputStream):
    """ Omit header information with hash as quote character """
    
    character = inputStream.read(1) 
    inputStream.seek(-1,1)
    while (character == b'#'):
        inputStream.readline() # ignore this line
        character = inputStream.read(1)
        inputStream.seek(-1,1)
        
        
def get_header_comments(inputStream):
    """ Return the header information """
    
    character = inputStream.read(1) 
    inputStream.seek(-1,1)
    header = b''
    while (character == b'#'):
        header = header + inputStream.readline()
        character = inputStream.read(1)
        inputStream.seek(-1,1)
        
    return header


def calculate_map(somWidth, somHeight, neurons, neuronWidth, neuronHeight, shareIntensity = False, border = 0, shape="box"):
    """ For quadratic map, it reads through the data and creates each neuron as a 1D array
        and then resizes it to the neuronSize print(neurons) """

    if shape == "box":
        neuronSize = np.array([neuronWidth, neuronHeight])
        mapSize = np.array([somWidth, somHeight])
        size = np.multiply(mapSize,np.array(neuronSize) + border) + border
        image = np.empty(size)
        image[:] = np.NAN
        for x in range(somWidth):
            for y in range(somHeight):
                if len(neurons[y + x*somHeight].shape)>0:
                    data = neurons[y + x*somHeight].reshape(neuronWidth, neuronHeight)
                    if not shareIntensity:
                        if np.max(data)-np.min(data) != 0:
                            data = 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))
                else:
                    data = np.ones((neuronWidth, neuronHeight)) * neurons[y + x*somHeight]
                image[border + x*(neuronWidth + border): (x+1) * (neuronWidth + border), border + y * (neuronHeight + border): (y+1) * (neuronHeight + border)] = data

    #For hexagonal map, it goes through each neuron while accounting for the hexagonal shape, starting at the bottom left corner
    if shape == "hex":
        size = np.multiply((somWidth, somHeight),np.array((neuronWidth, neuronHeight)) + border) + border
        size[1] = math.ceil(size[1] - (somHeight-1.0) * neuronHeight / 4.0)
        size[0] = math.ceil(size[0])
        size = np.asarray(size, int)
        image = np.empty(size)
        image[:] = np.NAN
        mapX = - np.floor((somWidth - 1) / 2)
        mapY = -1
        for neuron in neurons:
            mapY = mapY + 1
            if mapX < 0 and mapY > np.floor((somHeight - 1) / 2):
                mapX = mapX + 1
                mapY = - np.floor((somHeight - 1) / 2) - mapX
            elif mapX >= 0 and mapY > np.floor((somHeight - 1) / 2) - mapX:
                mapX = mapX + 1
                mapY = - np.floor((somHeight - 1) / 2)
            if mapY >= somHeight:
                print("abort")
                return image
            if len(neuron.shape) > 0 and not shareIntensity:
                if np.max(neuron)-np.min(neuron) != 0:
                    neuron = 1.0 * (neuron - np.min(neuron)) / (np.max(neuron) - np.min(neuron))
            for xPos in range(neuronWidth):
                for yPos in range(neuronHeight): 
                    # the corners are ignored
                    if math.floor(xPos/2.0 + yPos) < neuronHeight / 4.0 or \
                       math.floor(xPos/2.0 + yPos + 1) > neuronWidth + neuronHeight / 4.0 or \
                       math.floor(xPos/2.0 - yPos + 1) > neuronHeight / 4.0 or \
                       math.floor(xPos/2.0 - yPos) < -neuronWidth + neuronHeight / 4.0:
                        continue
                    # use the inner hex for plotting
                    else:
                        x = int(math.floor(xPos + ((somWidth - 1) / 2 + mapX + (mapY/2.0))* (neuronHeight+border) + border))
                        y = int(math.floor(yPos + ((somHeight - 1) / 2 + mapY) * (neuronHeight * 3.0/4.0 +border) + border))
                        if len(neuron.shape) > 0:
                            image[x][y] = neuron[yPos + xPos * neuronHeight]
                        else:
                            image[x][y] = neuron
    return image
