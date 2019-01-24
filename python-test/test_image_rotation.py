#!/usr/bin/env python3

"""
PINK Test image rotation
"""

__author__ = "Bernd Doser"
__email__ = "bernd.doser@h-its.org"
__license__ = "GPLv3"

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import pink
import scipy.misc

image = scipy.misc.ascent().astype(np.float32)

# min-max-normalization
image = (image - image.min()) / (image.max() - image.min())

print(image.shape)
print(image)

plt.axis("off")
plt.gray()
plt.imshow(image)
plt.show()

rotated_image = pink.rotate(image)
