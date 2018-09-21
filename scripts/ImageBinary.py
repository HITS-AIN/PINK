###SOM check to ensure SOMs are being printed correctly.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import struct

###Cat image
img=mpimg.imread('/lhome/lhopkinea/github/scripts/fuffi.jpg')
kitten=np.mean(img,axis=2)/255
plt.imshow(kitten)

###Square
square = np.zeros((100,100))
square[50:100,50:100] = 1
plt.imshow(square)

###Vertical lines
vlines = np.zeros((100,100))
vlines[:,10:20] = 1
vlines[:,30:40] = 1
vlines[:,50:60] = 1
vlines[:,70:80] = 1
vlines[:,90:100] = 1
plt.imshow(vlines)

###Horizontal
hlines = np.zeros((100,100))
hlines[10:20,:] = 1
hlines[30:40,:] = 1
hlines[50:60,:] = 1
hlines[70:80,:] = 1
hlines[90:100,:] = 1
plt.imshow(hlines)

###Triangle
triangle = np.zeros((100,100))
for i in range(20):
    triangle[i*5:(i+1)*5,0:(i+1)*5] = 1
plt.imshow(triangle)

###Another triangle
triangle2 = np.zeros((100,100))
for i in range(20):
    triangle2[i*5:(i+1)*5,int(100/2-i*2):int(100/2+i*2)] = 1
plt.imshow(triangle2)

###Another Square
square2 = np.zeros((100,100))
square2[0:50,0:50] = 1
plt.imshow(square2)

###Binary image with 6 images
output = open("/lhome/lhopkinea/github/scripts/testSOMInput.bin", 'wb') # output file opened for byte writing
output.write(struct.pack('i', 6)) # number of objects
output.write(struct.pack('i', 1)) # number of channels
output.write(struct.pack('i', 100)) # width
output.write(struct.pack('i', 100)) # height

#We transposed the images because tofile will only write as C order.
#We added the order=F to astype as a reminder that it's in Fortran order.
vlines.T.astype('f', order='F').tofile(output)
hlines.T.astype('f', order='F').tofile(output)
triangle.T.astype('f', order='F').tofile(output)
triangle2.T.astype('f', order='F').tofile(output)
square2.T.astype('f', order='F').tofile(output)
kitten.T.astype('f', order='F').tofile(output)

output.close()

###Quadratic SOM test
output = open("/lhome/lhopkinea/github/scripts/testSOMBox.bin", 'wb') # output file opened for byte writing
output.write(struct.pack('i', 1)) # number of channels
output.write(struct.pack('i', 2)) # SOM width
output.write(struct.pack('i', 3)) # SOM height
output.write(struct.pack('i', 1)) # SOM depth
output.write(struct.pack('i', 100)) # width
output.write(struct.pack('i', 100)) # height

kitten.T.astype('f', order='F').tofile(output)
vlines.T.astype('f', order='F').tofile(output)
hlines.T.astype('f', order='F').tofile(output)
triangle.T.astype('f', order='F').tofile(output)
triangle2.T.astype('f', order='F').tofile(output)
square2.T.astype('f', order='F').tofile(output)

output.close()

###Hexagonal SOM test
output = open("/lhome/lhopkinea/github/scripts/testSOMHex.bin", 'wb') # output file opened for byte writing
output.write(struct.pack('i', 1)) # number of channels
output.write(struct.pack('i', 3)) # SOM width
output.write(struct.pack('i', 3)) # SOM height
output.write(struct.pack('i', 1)) # SOM depth
output.write(struct.pack('i', 100)) # width
output.write(struct.pack('i', 100)) # height

kitten.T.astype('f', order='F').tofile(output)
vlines.T.astype('f', order='F').tofile(output)
hlines.T.astype('f', order='F').tofile(output)
triangle.T.astype('f', order='F').tofile(output)
triangle2.T.astype('f', order='F').tofile(output)
square2.T.astype('f', order='F').tofile(output)
square.T.astype('f', order='F').tofile(output)

output.close()
