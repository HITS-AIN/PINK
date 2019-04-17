import numpy
import math

def ignoreHeaderComments(file):
    """ Jump after header """

    file.seek(0)
    binary_start_position = 0
    for line in file:
        if line == b'# END OF HEADER\n':
            binary_start_position = file.tell()
            break

    file.seek(binary_start_position)

def calculateMap(somWidth, somHeight, neurons, neuronWidth, neuronHeight, shareIntensity = False, border = 0, shape="box"):
    #For quadratic map, it reads through the data and creates each neuron as a 1D array and then resizes it to the neuronSize
    #print(neurons)
    if shape == "box":
        neuronSize = numpy.array([neuronWidth, neuronHeight])
        mapSize = numpy.array([somWidth, somHeight])
        size = numpy.multiply(mapSize,numpy.array(neuronSize) + border) + border
        image = numpy.empty(size)
        image[:] = numpy.NAN
        for x in range(somWidth):
            for y in range(somHeight):
                if len(neurons[y + x*somHeight].shape)>0:
                    data = neurons[y + x*somHeight].reshape(neuronWidth, neuronHeight)
                    if not shareIntensity:
                        if numpy.max(data)-numpy.min(data) != 0:
                            data = 1.0 * (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
                else:
                    data = numpy.ones((neuronWidth, neuronHeight)) * neurons[y + x*somHeight]
                image[border + x*(neuronWidth + border): (x+1) * (neuronWidth + border), border + y * (neuronHeight + border): (y+1) * (neuronHeight + border)] = data

    #For hexagonal map, it goes through each neuron while accounting for the hexagonal shape, starting at the bottom left corner
    if shape == "hex":
        size = numpy.multiply((somWidth, somHeight),numpy.array((neuronWidth, neuronHeight)) + border) + border
        size[1] = math.ceil(size[1] - (somHeight-1.0) * neuronHeight / 4.0)
        size[0] = math.ceil(size[0])
        size = numpy.asarray(size, int)
        image = numpy.empty(size)
        image[:] = numpy.NAN
        mapX = - numpy.floor((somWidth - 1) / 2)
        mapY = -1
        for neuron in neurons:
            mapY = mapY + 1
            if mapX < 0 and mapY > numpy.floor((somHeight - 1) / 2):
                mapX = mapX + 1
                mapY = - numpy.floor((somHeight - 1) / 2) - mapX
            elif mapX >= 0 and mapY > numpy.floor((somHeight - 1) / 2) - mapX:
                mapX = mapX + 1
                mapY = - numpy.floor((somHeight - 1) / 2)
            if mapY >= somHeight:
                print("abort")
                return image
            if len(neuron.shape) > 0 and not shareIntensity:
                if numpy.max(neuron)-numpy.min(neuron) != 0:
                    neuron = 1.0 * (neuron - numpy.min(neuron)) / (numpy.max(neuron) - numpy.min(neuron))
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
